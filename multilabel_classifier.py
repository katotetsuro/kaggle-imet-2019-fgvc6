import argparse

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from PIL import Image
from os.path import join

import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
from sklearn.model_selection import KFold


num_classes = 1103


class MultilabelPandasDataset(chainer.dataset.dataset_mixin.DatasetMixin):
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
        one_hot_attributes = np.zeros(num_classes, dtype=np.int32)
        for a in attributes:
            one_hot_attributes[a] = 1
        image = Image.open(join(self.data_dir, image_id + '.png'))
        image = image.convert('RGB')
        image = np.asarray(image).astype(np.uint8)
        return image, one_hot_attributes


class ImgaugTransformer(chainer.datasets.TransformDataset):
    def __init__(self, size):
        self.seq = iaa.Sequential([
            iaa.Resize((size, size)),
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
            ])
        ])

    def __call__(self, in_data):
        x, t = in_data
        x = self.seq.augment_image(x)

        # to chainer style
        x = x.transpose(2, 0, 1).astype(np.float32) / 255.0
        return x, t


def get_dataset(size):
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    df = pd.read_csv('data/train.csv')
    train, test = next(iter(kf.split(df.index)))
    train = MultilabelPandasDataset(df.iloc[train], 'data/train')
    train = chainer.datasets.TransformDataset(train, ImgaugTransformer(size))
    test = MultilabelPandasDataset(df.iloc[test], 'data/train')
    test = chainer.datasets.TransformDataset(test, ImgaugTransformer(size))
    return train, test


class ResNet(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.res = chainer.links.model.vision.resnet.ResNet50Layers(
                pretrained_model=None)
            self.fc = chainer.links.Linear(None, num_classes)

    def forward(self, x):
        h = self.res.forward(x, layers=['pool5'])['pool5']
        h = self.fc(h)
        return h


class DebugModel(chainer.Chain):
    def __init__(self):
        print('using debug model')
        super().__init__()
        with self.init_scope():
            self.conv1 = chainer.links.Convolution2D(3, 64, ksize=3, stride=2)
            self.fc = chainer.links.Linear(None, num_classes)

    def forward(self, x):
        h = self.conv1(x)
        ksize = h.shape[2]
        h = F.average_pooling_2d(h, ksize)
        h = self.fc(h)
        return h


def f2_score(y, true):
    xp = chainer.backends.cuda.get_array_module(y)
    pred = y.array > 0.5
    tp = xp.sum(pred * true, axis=1)
    fp = xp.sum(pred * xp.logical_not(true), axis=1)
    fn = xp.sum(xp.logical_not(pred) * true, axis=1)
    p = tp/(tp+fp+1e-8)
    r = tp/(tp+fn+1e-8)
    f2 = 5*p*r/(4*p+r+1e-8)
    return xp.mean(p), xp.mean(r), xp.mean(f2)


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
        precision, recall, f2 = f2_score(y, t)
        chainer.reporter.report({'loss': loss,
                                 'precision': precision,
                                 'recall': recall,
                                 'f2': f2}, self)


def main():
    chainer.global_config.autotune = True
    chainer.cuda.set_max_workspace_size(512*1024*1024)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.01,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=300,
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
    args = parser.parse_args()

    print(args)
    # TODO save args

    train, test = get_dataset(args.size)
    model = DebugModel() if args.debug_model else ResNet()
    model = TrainChain(ResNet(), args.weight_positive_sample,
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
            if time > 1 * 60 * 60:
                print('時間切れで終了します。経過時間:{}'.format(time))
                return True
            return False

        def get_training_length(self):
            return self.epoch, 'epoch'

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu,
        converter=lambda batch, device: chainer.dataset.concat_examples(batch, device=device, padding=-1))
    trainer = training.Trainer(
        updater, TimeupTrigger(args.epoch), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu,
                                        eval_func=model.evaluate))

    if args.optimizer == 'sgd':
        trainer.extend(extensions.ExponentialShift(
            'lr', 0.5), trigger=(2, 'epoch'))

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
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/precision', 'validation/main/recall',
         'validation/main/f2', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    # predict


if __name__ == '__main__':
    main()
