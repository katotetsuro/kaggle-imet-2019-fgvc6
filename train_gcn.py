from os.path import join
from pathlib import Path
from glob import glob
import pickle
import argparse

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer.training import triggers
from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args
from chainercv.links.model.resnet import ResNet50, ResNet101
from chainercv.links.model.senet import SEResNet50

import numpy as np
import pandas as pd
from tqdm import tqdm

from adam import Adam
from lr_finder import LRFinder
from predict import ImgaugTransformer, ResNet, DebugModel, infer, backbone_catalog, num_attributes
from dataset import MultilabelPandasDataset, MixupDataset, count_cooccurrence, BalancedOrderSampler, calc_sampling_probs
from multilabel_classifier import make_folds, f2_score, get_dataset, find_optimal_threshold, set_random_seed, FScoreEvaluator, focal_loss, find_threshold
from graph_convolution import GraphConvolutionalNetwork
from cyclical_lr_scheduler import CosineAnnealing


def make_adjacent_matrix(train_file, t):
    df = pd.read_csv(train_file)
    counts = count_cooccurrence(df, num_attributes)
    conditional_probs = counts / np.diag(counts)
    binary_correlations = np.where(conditional_probs > t, 1.0, 0.0)
    p = 0.25
    a = np.maximum(np.sum(binary_correlations, axis=1) -
                   np.diag(binary_correlations), 1).reshape(-1, 1)
    reweighted_correlations = np.where(
        np.eye(len(binary_correlations)), 1, binary_correlations * p / a)

    reweighted_correlations = reweighted_correlations.astype(np.float32)

    # よくわからんけどソース読んだ結果
    D = np.power(np.sum(reweighted_correlations, axis=1), -0.5)
    D = np.diag(D)
    adj = reweighted_correlations.dot(D).T.dot(D)

    return adj


class GCNCNN(chainer.Chain):
    def __init__(self, adjacent, embeddings, train_cnn):
        super().__init__()
        with self.init_scope():
            self.gcn = GraphConvolutionalNetwork(adjacent)
            self.cnn = SEResNet50(pretrained_model='imagenet')
            self.cnn.pick = 'res5'

        self.embeddings = embeddings
        self.train_cnn = train_cnn

    def forward(self, x):
        if self.train_cnn:
            y_1 = self.cnn(x)
        else:
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                y_1 = self.cnn(x)

        y_1 = F.dropout(y_1)
        s = y_1.shape[2]
        y_1 = F.average_pooling_2d(y_1, s)[:, :, 0, 0]
        y_2 = self.gcn(self.embeddings)
        y_2 = F.dropout(y_2)

        h = F.matmul(y_1, y_2, transb=True)
        return h

    def freeze(self):
        if self.cnn.update_enabled:
            self.cnn.disable_update()

    def unfreeze(self):
        if not self.cnn.update_enabled:
            self.cnn.enable_update()

    def to_gpu(self, device=None):
        self.embeddings = chainer.backends.cuda.to_gpu(
            self.embeddings, device=device)
        return super().to_gpu(device)

    def to_cpu(self):
        self.embeddings = chainer.backends.cuda.to_cpu(
            self.embeddings, device=device)
        return super().to_cpu()


def multilabel_soft_margin_loss(y, t):
    loss = -(t * F.log(F.sigmoid(y)+1e-8) +
             (1 - t) * F.log(F.sigmoid(-y)+1e-8))
    return F.mean(loss)


class TrainChain(chainer.Chain):
    def __init__(self, model, loss_fn):
        super().__init__()
        with self.init_scope():
            self.model = model

        if loss_fn == 'focal':
            self.loss_fn = lambda x, t: F.sum(focal_loss(F.sigmoid(x), t))
        elif loss_fn == 'sigmoid':
            self.loss_fn = lambda x, t: F.sum(F.sigmoid_cross_entropy(
                x, t, reduce='no'))
        elif loss_fn == 'margin':
            self.loss_fn = multilabel_soft_margin_loss
        else:
            raise ValueError('unknown loss function. {}'.format(loss_fn))

    def loss(self, y, t):
        attribute_wise_loss = self.loss_fn(y, t)
        return attribute_wise_loss

    def forward(self, x, t):
        y = self.model(x)
        loss = self.loss(y, t)
        chainer.reporter.report(
            {'loss': loss}, self)
        return loss

    def freeze_extractor(self):
        self.model.freeze()

    def unfreeze_extractor(self):
        self.model.unfreeze()


def main(args=None):
    set_random_seed(63)
    chainer.global_config.autotune = True
    chainer.cuda.set_max_workspace_size(512*1024*1024)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.01,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=80,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--loss-function',
                        choices=['focal', 'sigmoid', 'margin'], default='focal')
    parser.add_argument(
        '--optimizer', choices=['sgd', 'adam', 'adabound'], default='adam')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--lr-search', action='store_true')
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument(
        '--backbone', choices=['resnet', 'seresnet', 'debug_model'], default='resnet')
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--find-threshold', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--feat', default='data/feat_original.npy', type=str)
    parser.add_argument('--adj-threshold', default=0.4, type=float)
    args = parser.parse_args() if args is None else parser.parse_args(args)

    print(args)

    if args.mixup and args.loss_function != 'focal':
        raise ValueError('mixupを使うときはfocal lossしか使えません（いまんところ）')

    train, test, balanced_sampler = get_dataset(
        args.data_dir, args.size, args.limit, args.mixup)
    adjacent = make_adjacent_matrix('data/train.csv', args.adj_threshold)
    embeddings = np.load(args.feat)
    base_model = GCNCNN(adjacent, embeddings, True)

    if args.pretrained:
        print('loading pretrained model: {}'.format(args.pretrained))
        chainer.serializers.load_npz(args.pretrained, base_model, strict=False)
    model = TrainChain(base_model, loss_fn=args.loss_function)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    if args.optimizer in ['adam', 'adabound']:
        optimizer = Adam(alpha=args.learnrate, adabound=args.optimizer ==
                         'adabound', weight_decay_rate=1e-5, gamma=5e-7)
    elif args.optimizer == 'sgd':
        optimizer = chainer.optimizers.MomentumSGD(lr=args.learnrate)

    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(10))
    # for p in base_model.cnn.params():
    #     if args.optimizer == 'adabound':
    #         p.update_rule.hyperparam.initial_alpha *= 0.1
    #     elif args.optimizer == 'adam':
    #         p.update_rule.hyperparam.alpha *= 0.1
    #     elif args.optimizer == 'sgd':
    #         p.update_rule.hyperparam.lr *= 0.1

    if not args.finetune:
        print('最初のエポックは特徴抽出層をfreezeします')
        model.freeze_extractor()

    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=8, n_prefetch=2, order_sampler=balanced_sampler)
    test_iter = chainer.iterators.MultithreadIterator(test, args.batchsize, n_threads=8,
                                                      repeat=False, shuffle=False)

    if args.find_threshold:
        # train_iter, optimizerなど無駄なsetupもあるが。。
        print('thresholdを探索して終了します')
        chainer.serializers.load_npz(
            join(args.out, 'bestmodel_loss'), base_model)
        print('lossがもっとも小さかったモデルに対しての結果:')
        find_threshold(base_model, test_iter, args.gpu, args.out)

        chainer.serializers.load_npz(
            join(args.out, 'bestmodel_f2'), base_model)
        print('f2がもっとも大きかったモデルに対しての結果:')
        find_threshold(base_model, test_iter, args.gpu, args.out)
        return

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu,
        converter=lambda batch, device: chainer.dataset.concat_examples(batch, device=device))
    trainer = training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(FScoreEvaluator(
        test_iter, model, device=args.gpu))

    if args.optimizer == 'sgd':
        # Adamにweight decayはあんまりよくないらしい
        optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(1e-4))
        # trainer.extend(extensions.ExponentialShift(
        #     'lr', 0.1), trigger=(args.epoch // 3, 'epoch'))
        trainer.extend(CosineAnnealing(
            lr_max=args.learnrate, T_0=1e+6), (1, 'iteration'))
        if args.lr_search:
            print('最適な学習率を探します')
            trainer.extend(LRFinder(1e-7, 1, 5, optimizer),
                           trigger=(1, 'iteration'))
    elif args.optimizer in ['adam', 'adabound']:
        if args.lr_search:
            print('最適な学習率を探します')
            trainer.extend(LRFinder(1e-7, 1, 5, optimizer,
                                    lr_key='alpha'), trigger=(1, 'iteration'))

        trainer.extend(extensions.ExponentialShift('alpha', 0.5),
                       trigger=triggers.EarlyStoppingTrigger(monitor='validation/main/loss'))

    # Take a snapshot of Trainer at each epoch
    trainer.extend(extensions.snapshot(
        filename='snaphot_epoch_{.updater.epoch}'), trigger=(10, 'epoch'))

    # Take a snapshot of Model which has best val loss.
    # Because searching best threshold for each evaluation takes too much time.
    trainer.extend(extensions.snapshot_object(
        model.model, 'bestmodel_loss'), trigger=triggers.MinValueTrigger('validation/main/loss'))
    trainer.extend(extensions.snapshot_object(
        model.model, 'bestmodel_f2'), trigger=triggers.MaxValueTrigger('validation/main/f2'))
    trainer.extend(extensions.snapshot_object(
        model.model, 'model_{.updater.epoch}'), trigger=(5, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(
        trigger=(args.log_interval, 'iteration')))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'lr', 'main/loss', 'validation/main/loss', 'validation/main/precision',
         'validation/main/recall', 'validation/main/f2', 'validation/main/threshold']))

    trainer.extend(extensions.ProgressBar(update_interval=args.log_interval))
    trainer.extend(extensions.observe_lr(), trigger=(
        args.log_interval, 'iteration'))
    trainer.extend(CommandsExtension())
    save_args(args, args.out)

    trainer.extend(lambda trainer: model.unfreeze_extractor(),
                   trigger=(1, 'epoch'))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # save args with pickle for prediction time
    pickle.dump(args, open(str(Path(args.out).joinpath('args.pkl')), 'wb'))

    # Run the training
    trainer.run()

    # find optimal threshold
    chainer.serializers.load_npz(join(args.out, 'bestmodel_loss'), base_model)
    print('lossがもっとも小さかったモデルに対しての結果:')
    find_threshold(base_model, test_iter, args.gpu, args.out)

    chainer.serializers.load_npz(join(args.out, 'bestmodel_f2'), base_model)
    print('f2がもっとも大きかったモデルに対しての結果:')
    find_threshold(base_model, test_iter, args.gpu, args.out)


if __name__ == '__main__':
    main()
