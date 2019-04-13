from os.path import join
from pathlib import Path
from glob import glob
import pickle
import argparse
from collections import defaultdict, Counter
import random
from itertools import combinations

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer.training import triggers
from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args

from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

from lr_finder import LRFinder
from predict import ImgaugTransformer, ResNet, DebugModel, infer, num_attributes, backbone_catalog


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


def make_folds(n_folds: int, df: pd.DataFrame) -> pd.DataFrame:
    """copyright https://github.com/lopuhin/kaggle-imet-2019/blob/master/imet/make_folds.py
    """
    cls_counts = Counter(cls for classes in df['attribute_ids'].str.split()
                         for cls in classes)
    fold_cls_counts = defaultdict(int)
    folds = [-1] * len(df)
    for item in tqdm(df.sample(frac=1, random_state=42).itertuples(),
                     total=len(df)):
        cls = min(item.attribute_ids.split(), key=lambda cls: cls_counts[cls])
        fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(n_folds)]
        min_count = min([count for _, count in fold_counts])
        random.seed(item.Index)
        fold = random.choice([f for f, count in fold_counts
                              if count == min_count])
        folds[item.Index] = fold
        for cls in item.attribute_ids.split():
            fold_cls_counts[fold, cls] += 1
    df['fold'] = folds
    return df


def count_cooccurrence(df):
    co = np.zeros((num_attributes, num_attributes), dtype=np.int32)
    for row in df.itertuples(index=False):
        id, attr = row
        for i, j in combinations(map(int, attr.split(' ')), 2):
            co[i, j] += 1
            co[j, i] += 1

    return co


def get_dataset(data_dir, size, limit):
    df = pd.read_csv(join(data_dir, 'train.csv'))
    co = count_cooccurrence(df)
    df = make_folds(5, df)
    train = df[df.fold != 0]
    test = df[df.fold == 0]
    train = train.drop(columns=['fold'])
    test = test.drop(columns=['fold'])
    if limit is not None:
        train = train[:limit]
        test = test[:limit]
    train = MultilabelPandasDataset(train, join(data_dir, 'train'))
    train = chainer.datasets.TransformDataset(
        train, ImgaugTransformer(size, True))
    test = MultilabelPandasDataset(test, join(data_dir, 'train'))
    test = chainer.datasets.TransformDataset(
        test, ImgaugTransformer(size, False))

    return train, test, co


def f2_score(pred, true):
    xp = chainer.backends.cuda.get_array_module(pred)
    tp = xp.sum(pred * true, axis=1)
    relevant = xp.sum(pred, axis=1)
    support = xp.sum(true, axis=1)
    p = tp/(relevant+1e-8)
    r = tp/(support+1e-8)
    f2 = 5*p*r/(4*p+r+1e-8)
    return xp.mean(p), xp.mean(r), xp.mean(f2)


def find_optimal_threshold(y, true):
    if isinstance(y, chainer.Variable):
        y = y.array
    y = chainer.backends.cuda.to_cpu(y)
    true = chainer.backends.cuda.to_cpu(true)

    f2_scores = np.array([f2_score(y > i, true)
                          for i in np.arange(0, 1, 0.01)])
    best_threshold_index = np.argmax(f2_scores[:, 2])
    best_threshold = best_threshold_index * 0.01
    return best_threshold, f2_scores[best_threshold_index]


def focal_loss(y_pred, y_true):
    """from https://www.kaggle.com/mathormad/pretrained-resnet50-focal-loss
    """
    gamma = 2.0
    epsilon = 1e-5
    pt = y_pred * y_true + (1-y_pred) * (1-y_true)
    pt = F.clip(pt, epsilon, 1-epsilon)
    CE = -F.log(pt)
    FL = (1-pt)**gamma * CE
    loss = F.sum(FL, axis=1)
    return loss


class TrainChain(chainer.Chain):
    def __init__(self, model, weight, loss_fn, cooccurrence, co_coef):
        super().__init__()
        with self.init_scope():
            self.model = model

        self.weight = weight
        if loss_fn == 'focal':
            self.loss_fn = lambda x, t: F.sum(focal_loss(F.sigmoid(x), t))
        elif loss_fn == 'sigmoid':
            self.loss_fn = lambda x, t: F.sum(F.sigmoid_cross_entropy(
                x, t, reduce='no'))
        else:
            raise ValueError('unknown loss function. {}'.format(loss_fn))

        self.cooccurrence = (cooccurrence == 0).astype(np.float32)
        # 対角成分はlossをかけない
        self.cooccurrence -= np.eye(self.cooccurrence.shape[0])
        self.co_coef = co_coef

    def loss(self, y, t):
        if isinstance(y, tuple):
            y, z = y
            second_stage_loss = self.loss_fn(z, t)
            z = F.sigmoid(z)
            two_stage = True
        else:
            #z = F.sigmoid(y)
            two_stage = False
        first_stage_loss = self.loss_fn(y, t)
        # xp = chainer.backends.cuda.get_array_module(t)
        # weights = xp.where(t == 0, 1, self.weight)
        # loss = F.mean(loss * weights)
        return first_stage_loss

    def forward(self, x, t):
        y = self.model(x)
        loss = self.loss(y, t)
        chainer.reporter.report(
            {'loss': loss/len(t)}, self)
        return loss

    # def evaluate(self, x, t):
    #     y = self.model(x)
    #     loss = self.loss(y, t)

    #     y = chainer.backends.cuda.to_cpu(F.sigmoid(y).array)
    #     t = chainer.backends.cuda.to_cpu(t)
    #     precision, recall, f2 = f2_score(y > 0.1, t)
    #     # threshold, (precision, recall, f2) = find_optimal_threshold(
    #     #     F.sigmoid(y), t)
    #     chainer.reporter.report({'loss': loss,
    #                              'precision': precision,
    #                              'recall': recall,
    #                              'f2': f2}, self)

    def freeze_extractor(self):
        self.model.freeze()

    def unfreeze_extractor(self):
        self.model.unfreeze()


def find_threshold(model, test_iter, gpu, out):
    if gpu >= 0:
        model.to_gpu()
    pred, true = infer(test_iter, model, gpu)
    threshold, scores = find_optimal_threshold(pred, true)
    print('しきい値:{} で F2スコア{}'.format(threshold, scores))
    np.save(open(str(Path(out).joinpath('thresholds.npy')), 'wb'), threshold)


class FScoreEvaluator(extensions.Evaluator):
    def evaluate(self):
        iterator = self._iterators['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = chainer.reporter.DictSummary()

        observation = {}
        target = self._targets['main']
        with chainer.reporter.report_scope(observation):
            with chainer.function.no_backprop_mode():
                pred, true, loss = infer(
                    it, target.model, self.device, target.loss_fn)
                threshold, (precision, recall,
                            f2) = find_optimal_threshold(pred, true)
                chainer.reporter.report({'loss': loss/it.batch_size,
                                         'precision': precision,
                                         'recall': recall,
                                         'f2': f2}, target)

        summary.add(observation)

        return summary.compute_mean()


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
    parser.add_argument('--weight-positive-sample',
                        '-w', type=float, default=1)
    parser.add_argument('--loss-function',
                        choices=['focal', 'sigmoid'], default='focal')
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--lr-search', action='store_true')
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument(
        '--backbone', choices=['resnet', 'seresnet', 'seresnext', 'debug_model'], default='resnet')
    parser.add_argument('--co-coef', type=float, default=4)
    parser.add_argument('--two-step', action='store_true')
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--find-threshold', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    args = parser.parse_args() if args is None else parser.parse_args(args)

    print(args)

    train, test, cooccurrence = get_dataset(
        args.data_dir, args.size, args.limit)
    base_model = backbone_catalog[args.backbone](args.dropout)

    if args.pretrained:
        print('loading pretrained model: {}'.format(args.pretrained))
        chainer.serializers.load_npz(args.pretrained, base_model, strict=False)
    model = TrainChain(base_model, args.weight_positive_sample,
                       loss_fn=args.loss_function, cooccurrence=cooccurrence, co_coef=args.co_coef)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    if args.optimizer == 'adam':
        optimizer = chainer.optimizers.Adam(alpha=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = chainer.optimizers.MomentumSGD(lr=args.learnrate)

    optimizer.setup(model)

    if not args.finetune:
        print('最初のエポックは特徴抽出層をfreezeします')
        model.freeze_extractor()

    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=8, n_prefetch=2)
    test_iter = chainer.iterators.MultithreadIterator(test, args.batchsize, n_threads=8,
                                                      repeat=False, shuffle=False)

    if args.find_threshold:
        # train_iter, optimizerなど無駄なsetupもあるが。。
        print('thresholdを探索して終了します')
        chainer.serializers.load_npz(join(args.out, 'bestmodel'), base_model)
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
        optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))
        trainer.extend(extensions.ExponentialShift(
            'lr', 0.5), trigger=(10, 'epoch'))
        if args.lr_search:
            print('最適な学習率を探します')
            trainer.extend(LRFinder(1e-7, 1, 5, optimizer),
                           trigger=(1, 'iteration'))
    elif args.optimizer == 'adam':
        if args.lr_search:
            print('最適な学習率を探します')
            trainer.extend(LRFinder(1e-7, 1, 5, optimizer,
                                    lr_key='alpha'), trigger=(1, 'iteration'))

        trainer.extend(extensions.ExponentialShift('alpha', 0.2),
                       trigger=triggers.EarlyStoppingTrigger(monitor='validation/main/loss'))

    # Take a snapshot of Trainer at each epoch
    trainer.extend(extensions.snapshot(
        filename='snaphot_epoch_{.updater.epoch}'), trigger=(10, 'epoch'))

    # Take a snapshot of Model which has best val loss.
    # Because searching best threshold for each evaluation takes too much time.
    trainer.extend(extensions.snapshot_object(
        model.model, 'bestmodel'), trigger=triggers.MinValueTrigger('validation/main/loss'))
    trainer.extend(extensions.snapshot_object(
        model.model, 'model_{.updater.epoch}'), trigger=(5, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(
        trigger=(args.log_interval, 'iteration')))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'lr', 'elapsed_time', 'main/loss', 'validation/main/loss', 'validation/main/precision',
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
    chainer.serializers.load_npz(join(args.out, 'bestmodel'), base_model)
    find_threshold(base_model, test_iter, args.gpu, args.out)


if __name__ == '__main__':
    main()
