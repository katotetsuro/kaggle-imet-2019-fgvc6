from os.path import join
from itertools import combinations
from collections import defaultdict, Counter
import random

import numpy as np
import pandas as pd
import chainer
from predict import num_attributes, backbone_catalog
from PIL import Image
from tqdm import tqdm

from predict import ImgaugTransformer


def count_cooccurrence(df, num_attributes=1103, count_self=True):
    if isinstance(df, str):
        df = pd.read_csv(df)
    co = np.zeros((num_attributes, num_attributes), dtype=np.int32)
    for row in df.itertuples(index=False):
        id, attr = row

        attrs = list(map(int, attr.split(' ')))
        for i, j in combinations(attrs, 2):
            co[i, j] += 1
            co[j, i] += 1

        if count_self:
            for i in attrs:
                co[i, i] += 1

    return co


def calc_sampling_probs(df, min_score=0, max_score=1):
    if isinstance(df, str):
        df = pd.read_csv(df)
    co = count_cooccurrence(df)
    freq = np.diag(co)
    score = np.clip(1 / freq, min_score, max_score) ** 4
    scores = []
    for row in df.itertuples(index=False):
        id, attr = row
        s = np.sum([score[i] for i in map(int, attr.split(' '))])
        scores.append(s)

    probs = scores / np.sum(scores)
    return probs


class BalancedOrderSampler(chainer.iterators.OrderSampler):
    def __init__(self, df):
        super().__init__()
        self.probs = calc_sampling_probs(df)

    def __call__(self, current_order, current_position):
        n = len(self.probs)
        return np.random.choice(n, n, replace=True, p=self.probs)


class MultilabelPandasDataset(chainer.dataset.DatasetMixin):
    def __init__(self, df, data_dir, dummy=False):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.dummy = dummy
        if dummy:
            print('画像をロードしないデータセットです')

    def __len__(self):
        return len(self.df)

    def get_example(self, i):
        image_id, attributes = self.df.iloc[i]
        attributes = list(map(int, attributes.split(' ')))
        # TODO one_hot vectorの保持はメモリの無駄か
        one_hot_attributes = np.zeros(num_attributes, dtype=np.int32)
        for a in attributes:
            one_hot_attributes[a] = 1

        if self.dummy:
            return np.zeros((32, 32, 3), dtype=np.float32), one_hot_attributes

        image = Image.open(join(self.data_dir, image_id + '.png'))
        image = image.convert('RGB')
        image = np.asarray(image).astype(np.uint8)
        return image, one_hot_attributes


class MixupDataset(chainer.dataset.DatasetMixin):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.indexes = np.arange(len(dataset))
        np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.dataset) // 2

    def get_example(self, i):
        img_1, label_1 = self.dataset[self.indexes[2*i]]
        img_2, label_2 = self.dataset[self.indexes[2*i+1]]
        alpha = 0.2
        mix_ratio = np.random.beta(alpha, alpha)
        img = mix_ratio * img_1 + (1-mix_ratio) * img_2
        label = mix_ratio * label_1 + (1-mix_ratio) * label_2

        if i == self.__len__() - 1:
            print('shuffle mixup dataset')
            np.random.shuffle(self.indexes)
        return img, label


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


def get_dataset(df_file, data_dir, size, limit, mixup):
    df = pd.read_csv(join(data_dir, df_file))
    df = make_folds(10, df)
    train = df[df.fold != 0]
    test = df[df.fold == 0]
    train = train.drop(columns=['fold'])
    test = test.drop(columns=['fold'])

    def flat_labels(df):
        arr = []
        for row in df.itertuples(index=False):
            id, attr = row
            arr.append(np.array([i for i in map(int, attr.split(' '))]))

        return np.concatenate(arr)

    not_included = np.where(np.bincount(flat_labels(
        train), minlength=num_attributes) == 0)[0]
    print('以下のラベルはtrainに含まれていません. ', not_included)

    for l in not_included:
        for row in test.itertuples(index=True):
            index, id, attr = row
            attrs = list(map(int, attr.split(' ')))
            if l in attrs:
                train = train.append(test.loc[index])
                test = test.drop(index)
                print(id, attr, 'が追加されました')
                break

    assert len(np.where(np.bincount(flat_labels(train),
                                    minlength=num_attributes) == 0)[0]) == 0

    if limit is not None:
        train = train[:limit]
        test = test[:limit]
        print('limitが指定されているときはShuffleOrderSamplerしか使えない')
        order_sampler = chainer.iterators.ShuffleOrderSampler()
    else:
        print('balanced order samplerをつかいます')
        order_sampler = BalancedOrderSampler(train)
    train = MultilabelPandasDataset(train, join(data_dir, 'train'))
    train = chainer.datasets.TransformDataset(
        train, ImgaugTransformer(size, True))

    test = MultilabelPandasDataset(test, join(data_dir, 'train'))
    test = chainer.datasets.TransformDataset(
        test, ImgaugTransformer(size, False))

    if mixup:
        print('mixup')
        train = MixupDataset(train)

    return train, test, order_sampler


class SubsetSampler(chainer.iterators.OrderSampler):
    def __init__(self, total_size, sample_size):
        self.total_size = total_size
        self.sample_size = sample_size

    def __call__(self, a, b):
        return np.random.choice(self.total_size, self.sample_size, replace=False)
