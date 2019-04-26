from os.path import join
import numpy as np
import chainer
from predict import num_attributes, backbone_catalog
from PIL import Image


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
