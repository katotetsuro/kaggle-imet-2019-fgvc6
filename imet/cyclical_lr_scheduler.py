import numpy as np
from warnings import warn
from chainer.training.extension import Extension


class CyclicalLRScheduler(Extension):
    def __init__(self):
        super().__init__()

    def __call__(self, trainer):
