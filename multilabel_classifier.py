import argparse

import chainer
import chainer.links as L
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
    def __init__(self):
        self.seq = iaa.Sequential([
            iaa.Resize((224, 224))
        ])

    def __call__(self, in_data):
        x, t = in_data
        x = self.seq.augment_image(x)

        # to chainer style
        x = x.transpose(2, 0, 1).astype(np.float32) / 255.0
        return x, t


def get_dataset():
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    df = pd.read_csv('data/train.csv')
    train, test = next(iter(kf.split(df.index)))
    train = MultilabelPandasDataset(df.iloc[train], 'data/train')
    train = chainer.datasets.TransformDataset(train, ImgaugTransformer())
    test = MultilabelPandasDataset(df.iloc[test], 'data/train')
    test = chainer.datasets.TransformDataset(test, ImgaugTransformer())
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
        h = chainer.functions.average_pooling_2d(h, ksize)
        h = self.fc(h)
        return h


def f2_score(y, true):
    xp = chainer.backends.cuda.get_array_module(y)
    pred = y.array > 0.5
    tp = xp.sum(pred * true, axis=1)
    fp = xp.sum(pred * xp.logical_not(true), axis=1)
    fn = xp.sum(xp.logical_not(pred) * true, axis=1)
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    f2 = 5*p*r/(4*p+r+1e-5)
    return xp.mean(p), xp.mean(r), xp.mean(f2)


class TrainChain(chainer.Chain):
    def __init__(self, model):
        super().__init__()
        with self.init_scope():
            self.model = model

    def forward(self, x, t):
        y = self.model(x)
        loss = chainer.functions.sigmoid_cross_entropy(y, t)
        chainer.reporter.report({'loss': loss}, self)
        return loss

    def evaluate(self, x, t):
        y = self.model(x)
        loss = chainer.functions.sigmoid_cross_entropy(y, t)
        precision, recall, f2 = f2_score(y, t)
        chainer.reporter.report({'loss': loss,
                                 'precision': precision,
                                 'recall': recall,
                                 'f2': f2}, self)


def main():
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
    parser.add_argument('--early-stopping', type=str,
                        help='Metric to watch for early stopping')
    parser.add_argument('--debug-model', action='store_true')
    args = parser.parse_args()

    print(args)
    # TODO save args

    train, test = get_dataset()
    model = DebugModel() if args.debug_model else ResNet()
    model = TrainChain(ResNet())
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))

    train_iter = chainer.iterators.MultithreadIterator(train, args.batchsize)
    test_iter = chainer.iterators.MultithreadIterator(test, args.batchsize,
                                                      repeat=False, shuffle=False)

    stop_trigger = (args.epoch, 'epoch')
    # Early stopping option
    if args.early_stopping:
        stop_trigger = triggers.EarlyStoppingTrigger(
            monitor=args.early_stopping, verbose=True,
            max_trigger=(args.epoch, 'epoch'))

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu,
        converter=lambda batch, device: chainer.dataset.concat_examples(batch, device=device, padding=-1))
    trainer = training.Trainer(updater, stop_trigger, out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu,
                                        eval_func=model.evaluate))

    # Reduce the learning rate by half every 25 epochs.
#    trainer.extend(extensions.ExponentialShift('lr', 0.5),
#                   trigger=(25, 'epoch'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(
        filename='snaphot_epoch_{.updater.epoch}'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
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


if __name__ == '__main__':
    main()
