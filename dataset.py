from os.path import join
from itertools import combinations
import numpy as np
import chainer
from predict import num_attributes, backbone_catalog
from PIL import Image


def count_cooccurrence(df, num_attributes=1103, count_self=True):
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
    co = count_cooccurrence(df)
    freq = np.diag(co)
    score = np.clip(1 / freq, min_score, max_score)
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
        return np.random.choice(n, n, replace=False, p=self.probs)


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
        # return np.zeros((32, 32, 3), dtype=np.float32), one_hot_attributes


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
