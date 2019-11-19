import numpy as np


class NormalNoise(object):
    def __init__(self, std):
        self.std = std

    def apply(self, noiseless_value):
        noisy_value = noiseless_value + np.random.normal(0, self.std, noiseless_value.shape)
        return noisy_value