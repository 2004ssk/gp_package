# means.py

import abc
import jax.numpy as jnp

class BaseMean(abc.ABC):
    @abc.abstractmethod
    def __call__(self, X, params):
        pass

class ZeroMean(BaseMean):
    def __call__(self, X, params):
        return jnp.zeros(X.shape[0])
