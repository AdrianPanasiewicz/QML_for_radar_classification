import numpy as np
from abc import ABC, abstractmethod

class BaseNoiseModel(ABC):
    def __init__(self):
        self.params_set_flag = False

    @abstractmethod
    def set_parameters(self, params):
        pass

    @abstractmethod
    def apply_noise(self, signal, signal_params):
        pass

class AdditiveWhiteGaussianNoise(BaseNoiseModel):
    def __init__(self, sigma=None):
        super().__init__()
        if sigma is not None:
            self.sigma = sigma
            self.params_set_flag = True

    def set_parameters(self, params):
        if "sigma" in params.keys():
            self.sigma = params["sigma"]
            self.params_set_flag=True
        else:
            raise ValueError("A key with sigma value is not present in params dictionary")

    def apply_noise(self, signal, signal_params=None):
        noise = (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)) * (self.sigma / np.sqrt(2))
        return signal+noise

    @classmethod
    def generate_noise(self, sigma, shape):
        noise = (np.random.randn(shape) + 1j * np.random.randn(shape)) * (sigma / np.sqrt(2))
        return noise