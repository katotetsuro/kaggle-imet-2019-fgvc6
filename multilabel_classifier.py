import argparse

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer.training import triggers
from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args

from PIL import Image
from os.path import join
from pathlib import Path
from glob import glob
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from lr_finder import LRFinder
from predict import ImgaugTransformer, ResNet, DebugModel, infer, num_attributes


class MultilabelPandasDataset(chainer.dataset.DatasetMixin):
    def __init__(self, df, data_dir):
        super().__init__()
        self.df = df
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def get_example(self, i):
        image_id, attributes = self.df.iloc[i]
        attributes = list(map(int, attributes.split(' ')))
        # TODO one_hot vectorの保持はメモリの無駄か
        one_hot_attributes = np.zeros(num_attributes, dtype=np.int32)
        for a in attributes:
            one_hot_attributes[a] = 1
        image = Image.open(join(self.data_dir, image_id + '.png'))
        image = image.convert('RGB')
        image = np.asarray(image).astype(np.uint8)
        return image, one_hot_attributes


def get_dataset(data_dir, size, limit):
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    df = pd.read_csv(join(data_dir, 'train.csv'))
    train, test = next(iter(kf.split(df.index)))
    if limit is not None:
        train = train[:limit]
        test = test[:limit]
    train = MultilabelPandasDataset(df.iloc[train], join(data_dir, 'train'))
    train = chainer.datasets.TransformDataset(
        train, ImgaugTransformer(size, True))
    test = MultilabelPandasDataset(df.iloc[test], join(data_dir, 'train'))
    test = chainer.datasets.TransformDataset(
        test, ImgaugTransformer(size, False))
    return train, test


def f2_score(pred, true):
    xp = chainer.backends.cuda.get_array_module(pred)
    tp = xp.sum(pred * true, axis=1)
    relevant = xp.sum(pred, axis=1)
    support = xp.sum(true, axis=1)
    p = tp/(relevant+1e-8)
    r = tp/(support+1e-8)
    f2 = 5*p*r/(4*p+r+1e-8)
    return xp.mean(p), xp.mean(r), xp.mean(f2)


def find_optimal_threshold(y, true, per_attribute_search):
    if isinstance(y, chainer.Variable):
        y = y.array
    # num_attributesの数だけ独立にしきい値を探していく
    y = chainer.backends.cuda.to_cpu(y)
    true = chainer.backends.cuda.to_cpu(true)
    thresholds = np.zeros(
        num_attributes if per_attribute_search else 1, dtype=np.float32)
    step = 0.01

    for j in range(len(thresholds)):
        f2_scores = []
        for i in np.arange(0, 1, step):
            thresholds[j] = i
            s = f2_score(y > thresholds, true)
            f2_scores.append(s)

        f2_scores = np.asarray(f2_scores)
        best_threshold_index = np.argmax(f2_scores[:, 2])
        best_threshold = best_threshold_index * step
        thresholds[j] = best_threshold

    mean_threshold = np.mean(thresholds)
    return thresholds, mean_threshold, f2_scores[best_threshold_index]


def focal_loss(logit, y_true):
    """from https://www.kaggle.com/mathormad/pretrained-resnet50-focal-loss
    """
    gamma = 2.0
    epsilon = 1e-5
    y_pred = F.sigmoid(logit)
    pt = y_pred * y_true + (1-y_pred) * (1-y_true)
    pt = F.clip(pt, epsilon, 1-epsilon)
    CE = -F.log(pt)
    FL = (1-pt)**gamma * CE
    loss = F.mean(F.sum(FL, axis=1))
    return loss


class TrainChain(chainer.Chain):
    def __init__(self, model, weight, loss_fn):
        super().__init__()
        with self.init_scope():
            self.model = model

        self.weight = weight
        if loss_fn == 'focal':
            self.loss_fn = focal_loss
        elif loss_fn == 'sigmoid':
            self.loss_fn = lambda x, t: F.sigmoid_cross_entropy(
                x, t, reduce='no')
        else:
            raise ValueError('unknown loss function. {}'.format(loss_fn))

    def loss(self, y, t):
        loss = self.loss_fn(y, t)
        xp = chainer.backends.cuda.get_array_module(t)
        weights = xp.where(t == 0, 1, self.weight)
        loss = F.mean(loss * weights)
        return loss

    def forward(self, x, t):
        y = self.model(x)
        loss = self.loss(y, t)
        chainer.reporter.report({'loss': loss}, self)
        return loss

    def evaluate(self, x, t):
        y = self.model(x)
        loss = self.loss_fn(y, t)
        y = F.sigmoid(y)
        thresholds, mean_threshold, (precision,
                                     recall, f2) = find_optimal_threshold(y, t, False)
        chainer.reporter.report({'loss': loss,
                                 'precision': precision,
                                 'recall': recall,
                                 'f2': f2,
                                 'threshold': mean_threshold}, self)


def main(args=None):
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
    parser.add_argument('--debug-model', action='store_true')
    parser.add_argument('--weight-positive-sample',
                        '-w', type=float, default=1)
    parser.add_argument('--loss-function',
                        choices=['focal', 'sigmoid'], default='focal')
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--hour', type=int, default=6)
    parser.add_argument('--lr-search', action='store_true')
    args = parser.parse_args() if args is None else parser.parse_args(args)

    print(args)

    train, test = get_dataset(args.data_dir, args.size, args.limit)
    base_model = DebugModel() if args.debug_model else ResNet()
    model = TrainChain(base_model, args.weight_positive_sample,
                       loss_fn=args.loss_function)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    if args.optimizer == 'adam':
        optimizer = chainer.optimizers.Adam()
    elif args.optimizer == 'sgd':
        optimizer = chainer.optimizers.MomentumSGD(lr=args.learnrate)

    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))

    train_iter = chainer.iterators.MultithreadIterator(
        train, args.batchsize, n_threads=8)
    test_iter = chainer.iterators.MultithreadIterator(test, args.batchsize, n_threads=8,
                                                      repeat=False, shuffle=False)

    class TimeupTrigger():
        def __init__(self, epoch):
            self.epoch = epoch

        def __call__(self, trainer):
            epoch = trainer.updater.epoch
            if epoch > args.epoch:
                return True
            time = trainer.elapsed_time
            if time > args.hour * 60 * 60:
                print('時間切れで終了します。経過時間:{}'.format(time))
                return True
            return False

        def get_training_length(self):
            return self.epoch, 'epoch'

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu,
        converter=lambda batch, device: chainer.dataset.concat_examples(batch, device=device))
    trainer = training.Trainer(
        updater, TimeupTrigger(args.epoch), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu,
                                        eval_func=model.evaluate))

    if args.optimizer == 'sgd':
        trainer.extend(extensions.ExponentialShift(
            'lr', 0.5), trigger=(2, 'epoch'))
        if args.lr_search:
            print('最適な学習率を探します')
            trainer.extend(LRFinder(1e-7, 1, 5, optimizer),
                           trigger=(1, 'iteration'))
    elif args.optimizer == 'adam':
        if args.lr_search:
            print('最適な学習率を探します')
            trainer.extend(LRFinder(1e-7, 1, 5, optimizer,
                                    lr_key='alpha'), trigger=(1, 'iteration'))

    # Take a snapshot of Trainer at each epoch
    trainer.extend(extensions.snapshot(
        filename='snaphot_epoch_{.updater.epoch}'))

    # Take a snapshot of Model which has best F2 score.
    trainer.extend(extensions.snapshot_object(
        model.model, 'bestmodel'), trigger=triggers.MaxValueTrigger('validation/main/f2'))
    # MaxValueTriggerがちゃんと使えてるか確認できるまで、毎エポック保存する
    trainer.extend(extensions.snapshot_object(
        model.model, 'model_{.updater.epoch}'), trigger=(1, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'lr', 'elapsed_time', 'main/loss', 'validation/main/loss',
         'validation/main/precision', 'validation/main/recall',
         'validation/main/f2', 'validation/main/threshold']))

    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.observe_lr(), trigger=(100, 'iteration'))
    trainer.extend(CommandsExtension())
    save_args(args, args.out)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    # save args with pickle for prediction time
    pickle.dump(args, open(str(Path(args.out).joinpath('args.pkl')), 'wb'))

    # find optimal threshold
    base_model = DebugModel() if args.debug_model else ResNet()
    chainer.serializers.load_npz(join(args.out, 'bestmodel'), base_model)
    if args.gpu >= 0:
        model.to_gpu()

    pred, true = infer(test_iter, base_model, args.gpu)
    thresholds, mean_threshold, scores = find_optimal_threshold(
        pred, true, True)
    print('しきい値:{} で F2スコア{}'.format(mean_threshold, scores))
    thresholds = chainer.backends.cuda.to_cpu(thresholds)
    np.save(open(Path(args.out).joinpath('thresholds.npy'), 'wb'), thresholds)


if __name__ == '__main__':
    main()
