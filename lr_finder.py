import numpy as np
from warnings import warn
from chainer.training.extension import Extension


class LRFinder(Extension):
    def __init__(self, min_lr, max_lr, factor, optimizer, loss_key='main/loss', lr_key='lr'):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.factor = factor
        self.optimizer = optimizer
        self.loss_key = loss_key
        self.lr_key = lr_key

        setattr(self.optimizer, self.lr_key, min_lr)
        self.losses = []
        self.active = True

    def _set_lr(self, lr):
        setattr(self.optimizer, self.lr_key, lr)

    def _get_lr(self):
        return getattr(self.optimizer, self.lr_key)

    def __call__(self, trainer):
        if not self.active:
            return

        loss = trainer.observation[self.loss_key].array
        current_lr = self._get_lr()
        print('lr_finder observed loss: {}, current lr={}'.format(loss, current_lr))
        self.losses.append(loss)

        if current_lr < self.max_lr:
            self._set_lr(current_lr * self.factor)
            return

        # determine best learning rate
        losses = np.asarray(self.losses)
        diffs = losses[1:] - losses[:-1]
        indexes = np.argsort(losses)
        if indexes[0] == 0 or indexes[0] == len(indexes)-1:
            warn(
                'minimum value of loss has observed at boundary. It might be better to test wider range')

        for i in indexes:
            if i == 0 or i == len(indexes) - 1:
                continue

            if diffs[i-1] < 0 and diffs[i] < 0:
                best_lr = self.min_lr * self.factor ** i
                print('best_lr:{}'.format(best_lr))
                self._set_lr(best_lr)
                self.active = False
                break

        if self.active:
            warn('cannot find good LR. Range might be too small or factor is too large. number of observed={}'.format(
                len(losses)))
            self.active = False
