import pickle
from argparse import ArgumentParser
from pathlib import Path
from os.path import join
from glob import glob
from math import sqrt

from PIL import Image
import chainer
import chainer.functions as F
from imgaug import augmenters as iaa
import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    from chainercv.links.model.senet import SEResNet50, SEResNeXt50
    from chainercv.links.model.resnet import ResNet50
except ImportError:
    print('kaggle kernelではchainercvを直接importできないので、dillでロードする')
    import dill
    SEResNet50 = dill.load(
        open('../input/model-definitions/seresnet50_def.dill', 'rb'))
    SEResNeXt50 = dill.load(
        open('../input/model-definitions/seresnext50_def.dill', 'rb'))
    ResNet50 = dill.load(
        open('../input/model-definitions/resnet50_def.dill', 'rb'))

num_attributes = 1103

# copyright https://github.com/lopuhin/kaggle-imet-2019/blob/master/imet/utils.py
import os
ON_KAGGLE: bool = 'KAGGLE_WORKING_DIR' in os.environ


class ImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, image_files):
        super().__init__()
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def get_example(self, i):
        image = Image.open(self.image_files[i])
        image = image.convert('RGB')
        image = np.asarray(image).astype(np.uint8)
        return image,


class ImgaugTransformer(chainer.datasets.TransformDataset):
    def __init__(self, size, train):
        self.seq = iaa.Sequential([iaa.OneOf([
            iaa.CropToFixedSize(size, size),
            iaa.Resize((size, size))
        ]),  # end of OneOf
            iaa.Fliplr(0.5),
            iaa.PerspectiveTransform(0.01)])

        if train:
            self.seq.append(iaa.CoarseSaltAndPepper(0.2, size_percent=0.01))

        self.mean = np.array([123.15163084, 115.90288257, 103.0626238],
                             dtype=np.float32).reshape(3, 1, 1)

    def __call__(self, in_data):
        if len(in_data) == 2:
            x, t = in_data
            x = self.seq.augment_image(x)
            # to chainer style
            x = x.transpose(2, 0, 1).astype(np.float32)
            x -= self.mean
            return x, t
        elif len(in_data) == 1:
            x, = in_data
            x = self.seq.augment_image(x)
            # to chainer style
            x = x.transpose(2, 0, 1).astype(np.float32)
            x -= self.mean
            return x,


class ResNet(chainer.Chain):
    def __init__(self, dropout):
        super().__init__()
        with self.init_scope():
            self.res = ResNet50(
                pretrained_model=None if ON_KAGGLE else 'imagenet')
            self.res.pick = 'pool5'
            self.fc = chainer.links.Linear(
                None, num_attributes, initialW=chainer.initializers.uniform.Uniform(sqrt(1/2048)))

        self.dropout = dropout

    @chainer.static_graph
    def forward(self, x):
        h = self.res(x)
        if self.dropout:
            h = F.dropout(h)
        h = self.fc(h)
        return h


class SEResNet(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.res = SEResNet50(
                pretrained_model=None if ON_KAGGLE else 'imagenet')
            self.res.pick = 'pool5'
            self.fc = chainer.links.Linear(
                None, num_attributes, initialW=chainer.initializers.uniform.Uniform(sqrt(1/2048)))

    @chainer.static_graph
    def forward(self, x):
        h = self.res(x)
        h = self.fc(h)
        return h


class SEResNeXt(chainer.Chain):
    """ResNetクラスとほとんどかわらないんだけど、SEResNextはstatic_graphにできないっぽい？
    """

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.res = SEResNeXt50(
                pretrained_model=None if ON_KAGGLE else 'imagenet')
            self.res.pick = 'pool5'
            self.fc1 = chainer.links.Linear(None, 512)
            self.fc2 = chainer.links.Linear(None, num_attributes)

    def forward(self, x):
        h = self.res(x)
        h = self.fc1(h)
        # ここにh.unchainを入れる方が最初のエポックはめちゃくちゃ早くなる
        h = F.relu(h)
        h = F.dropout(h)
        h = self.fc2(h)
        return h


class DebugModel(chainer.Chain):
    def __init__(self):
        print('using debug model')
        super().__init__()
        with self.init_scope():
            self.conv1 = chainer.links.Convolution2D(3, 64, ksize=3, stride=2)
            self.fc = chainer.links.Linear(None, num_attributes)

    @chainer.static_graph
    def forward(self, x):
        h = self.conv1(x)
        ksize = h.shape[2]
        h = F.average_pooling_2d(h, ksize)
        h = self.fc(h)
        return h


backbone_catalog = {
    'debug_model': DebugModel,
    'resnet': ResNet,
    'seresnet': SEResNet,
    'seresnext': SEResNeXt
}


def infer(data_iter, model, gpu):
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        data_iter.reset()
        pred = []
        true = []
        for batch in data_iter:
            batch = chainer.dataset.concat_examples(batch, device=gpu)
            if len(batch) == 2:
                x, t = batch
                true.append(chainer.backends.cuda.to_cpu(t))
            else:
                x, = batch
            y = model(x)
            if isinstance(y, tuple):
                y = y[1]
            y = F.sigmoid(y)
            pred.append(chainer.backends.cuda.to_cpu(y.array))
    pred = np.concatenate(pred)
    if len(true):
        true = np.concatenate(true)
        return pred, true
    else:
        return pred


def main(_args=None):
    parser = ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('tta', type=int, default=4)
    parser.add_argument('threshold', type=float, default=0.28)
    launch_args = parser.parse_args() if _args is None else parser.parse_args(
        _args)
    pretrained_model_dir = launch_args.dir

    # saved args which used in training phase
    args = pickle.load(
        open(Path(pretrained_model_dir).joinpath('args.pkl'), 'rb'))

    # test用画像の場所は決め打ちする
    if _args is not None:
        print('kaggle environment detected. overwriting data_dir...')
        args.data_dir = '../input/imet-2019-fgvc6'

    base_model = backbone_catalog[args.backbone](args.dropout)
    chainer.serializers.load_npz(
        join(pretrained_model_dir, 'bestmodel'), base_model)
    if args.gpu >= 0:
        # kaggleはGPU一個だけ
        chainer.backends.cuda.get_device_from_id(0).use()
        base_model.to_gpu()

    best_threshold = launch_args.threshold
    image_files = glob(join(args.data_dir, 'test/*.png'))
    test = ImageDataset(image_files)
    test = chainer.datasets.TransformDataset(
        test, ImgaugTransformer(args.size, launch_args.tta > 1))
    test_iter = chainer.iterators.MultiprocessIterator(
        test, args.batchsize, repeat=False, shuffle=False, n_processes=8, n_prefetch=2)

    preds = []
    for _ in tqdm(range(launch_args.tta), total=launch_args.tta):
        print('tta')
        pred = infer(test_iter, base_model, args.gpu)
        preds.append(pred)

    pred = np.mean(preds, axis=0)
    indexes = np.argsort(pred, axis=1)[:, ::-1][:, :15]
    attributes = []
    for i, p in zip(indexes, pred):
        attr = i[p[i] > best_threshold]
        attr = map(str, attr)
        attributes.append(' '.join(attr))

    submit_df = pd.DataFrame()
    submit_df['id'] = [Path(p).stem for p in image_files]
    submit_df['attribute_ids'] = attributes
    submit_df.to_csv('submission.csv', index=False)
    print(submit_df.head(5))


if __name__ == '__main__':
    main()
#    main(['../input/xxxxx', '4', '0.28'])
