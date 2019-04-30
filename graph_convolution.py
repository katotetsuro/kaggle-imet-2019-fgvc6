import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np

from chainercv.links.model.resnet import ResNet50


# class GraphModule(chainer.Chain):
#     def __init__(self, adjacent, out_dim):
#         super().__init__()
#         self.adjacent = adjacent
#         with self.init_scope():
#             self.w_in = L.Linear(None, out_dim)
#             self.w_out = L.Linear(None, out_dim)
#             self.w_loop = L.Linear(None, out_dim)

#     def forward(self, x):
#         xp = chainer.backends.cuda.get_array_module(x)
#         y_loop = self.w_loop(x)

#         def conv(w, m):
#             return F.mean(w(x[m])) if (m == True).any() else chainer.Variable(xp.zeros(1, dtype=np.float32)).reshape()

#         ys = []
#         for n in range(len(x)):
#             out_mask = self.adjacent[n, :]
#             y = conv(self.w_out, out_mask)
#             in_mask = self.adjacent[:, n]
#             y += conv(self.w_in, in_mask)
#             ys.append(y + y_loop[n])

#         ys = F.vstack(ys)
#         ys = F.sigmoid(ys)
#         return ys

class GraphModule(chainer.Chain):
    def __init__(self, adjacent, out_dim, nobias):
        super().__init__()
        w = chainer.initializers.Uniform(scale=1/np.sqrt(2048))
        with self.init_scope():
            self.w = L.Linear(None, out_dim, nobias=nobias, initialW=w)

        self.adjacent = adjacent

    def forward(self, x):
        return F.matmul(self.adjacent, self.w(x))

    def to_gpu(self, device=None):
        super().to_gpu(device)
        chainer.backends.cuda.to_gpu(self.adjacent, device=device)

    def to_cpu(self):
        super().to_cpu()
        chainer.backends.cuda.to_cpu(self.adjacent, device=device)


class GraphConvolutionalNetwork(chainer.Chain):
    def __init__(self, adjacent):
        super().__init__()
        self.adjacent = adjacent
        with self.init_scope():
            self.gcn = chainer.Sequential(
                GraphModule(adjacent, 1024, True),
                F.leaky_relu,
                GraphModule(adjacent, 2048, True))

    def forward(self, x):
        return self.gcn(x)


# class GCNMultilabelPredictor(chainer.Chain):
#     def __init__(self, adjacent):
#         super().__init__()
#         with self.init_scope():
#             self.cnn = ResNet50()
#             self.gcn = GraphConvolutionalNetwork(adjacent)

#         self.train_cnn = False

#     def forward(self, x):
#         if self.train_cnn:
#             with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
#                 image_feature = self.cnn(x)
#         else:
#             image_feature = self.cnn(x)

#         graph_weight = self.gcn()
