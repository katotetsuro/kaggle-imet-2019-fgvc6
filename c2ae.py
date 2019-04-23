import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


class C2AETrainChain(chainer.Chain):
    """
    copyright https://github.com/hinanmu/C2AE_tensorflow/blob/master/src/network.py
    """

    def __init__(self, model):
        super().__init__()
        with self.init_scope():
            self.model = model

    def pairwise_and(self, a, b):
        return a[:, :, None] & b[:, None, :]

    def pairwise_sub(self, a, b):
        return a[:, :, None] - b[:, None, :]

    # def embedding_loss(self, x, e):
    #     xp = chainer.backends.cuda.get_array_module(x)
    #     I = xp.eye(x.shape[1])
    #     C1 = (x - e)  # shape (batchsize, embedding_dim)
    #     C1 = F.matmul(C1, C1, transb=True)  # (embedding_dim, embedding_dim)

    #     def trace(x):
    #         return F.sum(F.diagonal(x))

    #     C1 = trace(C1)

    #     C2 = F.matmul(x, x, transa=True) - I
    #     C2 = F.matmul(C2, C2, transb=True)

    #     C3 = F.matmul(e, e, transa=True) - I
    #     C3 = F.matmul(C3, C3, transb=True)
    #     loss = C1 + 0.5 * trace(C2 + C3)
    #     return loss

    def embedding_loss(self, x, e):
        xp = chainer.backends.cuda.get_array_module(x)
        I = xp.eye(x.shape[1])
        C1 = F.sum((x-e)**2)
        bs = len(x)
        C2 = F.sum((F.matmul(x, x, transa=True) / bs - I)**2)
        C3 = F.sum((F.matmul(e, e, transa=True) / bs - I)**2)
        loss = C1 + 0.5 * (C2 + C3)
        return loss

    def output_loss(self, predictions, labels):
        """Computational error function,k属于Y，l属于Y补，计算ck - cl值，此时误差是对称的
        Parameters
        ----------
        y : tensorflow tensor {0,1}
            binary indicator matrix with label assignments.
        output : tensorflow tensor [0,1]
            neural network output value
        Returns
        -------
        tensorflow tensor
        """

        y_i = labels == 1
        y_not_i = labels == 0

        # get indices to check
        truth_matrix = self.pairwise_and(y_i, y_not_i).astype(np.float32)

        # calculate all exp'd differences
        # through and with truth_matrix, we can get all c_i - c_k(appear in the paper)
        sub_matrix = self.pairwise_sub(predictions, predictions)
        exp_matrix = F.exp(-(5 * sub_matrix))

        # check which differences to consider and sum them
        sparse_matrix = exp_matrix * truth_matrix
        sums = F.sum(sparse_matrix, axis=(1, 2))

        # get normalizing terms and apply them
        y_i_sizes = F.sum(y_i.astype(np.float32), axis=1)
        y_i_bar_sizes = F.sum(y_not_i.astype(np.float32), axis=1)
        normalizers = y_i_sizes * y_i_bar_sizes

        loss = sums / (5*normalizers)
        loss = F.clip(loss, -1e+6, 1e+6)
        loss = F.sum(loss)
        return loss

    def loss(self, encoded_x, decoded_x, t):
        encoded_l, decoded_l = self.model.encode_decode_label(
            t.astype(np.float32))
        e_loss = self.embedding_loss(encoded_x, encoded_l)
        o_loss = self.output_loss(decoded_l, t)
        return e_loss, o_loss * 10

    def forward(self, x, t):
        encoded_x, decoded_x = self.model(x)
        e_loss, o_loss = self.loss(encoded_x, decoded_x, t)
        loss = e_loss + o_loss
        chainer.reporter.report(
            {'embed_loss': e_loss,
             'output_loss': o_loss,
             'loss': loss}, self)
        return loss

    def freeze_extractor(self):
        self.model.freeze()

    def unfreeze_extractor(self):
        self.model.unfreeze()
