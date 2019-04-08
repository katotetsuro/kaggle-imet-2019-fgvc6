import pickle
from argparse import ArgumentParser
from pathlib import Path
from os.path import join
from glob import glob
from PIL import Image

import chainer
import chainer.functions as F
from imgaug import augmenters as iaa
import pandas as pd
import numpy as np

try:
    from chainercv.links.model.senet import SEResNeXt50
except ImportError:
    print('kaggle kernelではchainercvを直接importできないので、dillでロードする')

num_attributes = 1103


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
        self.seq = iaa.Sequential([
            iaa.Resize((size, size))
        ])
        if train:
            self.seq.append(iaa.OneOf([
                iaa.Crop((0, 50)),
                iaa.Fliplr(0.5),
                iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-20, 20),
                )
            ]))

        self.mean = np.array([0.485, 0.456, 0.406],
                             dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225],
                            dtype=np.float32).reshape(3, 1, 1)

    def __call__(self, in_data):
        if len(in_data) == 2:
            x, t = in_data
            x = self.seq.augment_image(x)
            # to chainer style
            x = x.transpose(2, 0, 1).astype(np.float32) / 255.0
            x = (x - self.mean) / self.std
            return x, t
        elif len(in_data) == 1:
            x, = in_data
            x = self.seq.augment_image(x)
            # to chainer style
            x = x.transpose(2, 0, 1).astype(np.float32) / 255.0
            x = (x - self.mean) / self.std
            return x,


class ResNet(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.res = chainer.links.model.vision.resnet.ResNet50Layers(
                pretrained_model='auto')
            self.fc1 = chainer.links.Linear(None, 512)
            self.fc2 = chainer.links.Linear(None, num_attributes)

    @chainer.static_graph
    def forward(self, x):
        h = self.res.forward(x, layers=['pool5'])['pool5']
        h = self.fc1(h)
        h = F.relu(h)
        h = F.dropout(h)
        h = self.fc2(h)
        return h


class SEResNeXt(chainer.Chain):
    """ResNetクラスとほとんどかわらないんだけど、SEResNextはstatic_graphにできないっぽい？
    """

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.res = SEResNeXt50()
            self.res.pick = 'pool5'
            self.fc1 = chainer.links.Linear(None, 512)
            self.fc2 = chainer.links.Linear(None, num_attributes)

    def forward(self, x):
        h = self.res(x)
        h = self.fc1(h)
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
            y = F.sigmoid(y)
            pred.append(chainer.backends.cuda.to_cpu(y.array))
    pred = np.concatenate(pred)
    if len(true):
        true = np.concatenate(true)
        return pred, true
    else:
        return pred


def main():
    parser = ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('tta', type=int, default=4)
    current_args = parser.parse_args()
    data_dir = current_args.dir
    args = pickle.load(open(Path(data_dir).joinpath('args.pkl'), 'rb'))

    # pretrained modelの場所、test用画像の場所は決め打ちする
    # args.data_dir = '../input'

    base_model = backbone_catalog[args.backbone]()
    chainer.serializers.load_npz(join(data_dir, 'bestmodel'), base_model)

    best_threshold = np.load(join(data_dir, 'thresholds.npy'))
    image_files = glob(join(args.data_dir, 'test/*.png'))
    test = ImageDataset(image_files)
    test = chainer.datasets.TransformDataset(
        test, ImgaugTransformer(args.size, True))
    test_iter = chainer.iterators.MultithreadIterator(
        test, args.batchsize, repeat=False, shuffle=False, n_threads=8)

    preds = []
    for _ in range(current_args.tta):
        pred = infer(test_iter, base_model, args.gpu)
        preds.append(pred)

    pred = np.mean(preds, axis=0)
    indexes = np.argsort(pred, axis=1)[:, ::-1][:, :10]
    attributes = []
    for i, p in zip(indexes, pred):
        attr = i[p[i] > best_threshold]
        attr = map(str, attr)
        attributes.append(' '.join(attr))

    submit_df = pd.DataFrame()
    submit_df['id'] = [Path(p).stem for p in image_files]
    submit_df['attribute_ids'] = attributes
    submit_df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
