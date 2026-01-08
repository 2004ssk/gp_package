# kernels.py

import abc
import jax.numpy as jnp
from jax import vmap

class BaseKernel(abc.ABC):
    """
    Abstract class for a kernel
    """

    @abc.abstractmethod
    def __call__(self, x1, x2, params):
        """
        Computes covariance between x1 and x2 given kernel parameter
        """
        pass

class RBFKernel(BaseKernel):
    """
    Radial Basis Function 
    """
    def __call__(self, x1, x2, params):
        length_scale = params["length_scale"]
        variance = params["variance"]
        sqdist = jnp.sum((x1 - x2)**2)
        return variance * jnp.exp(-0.5 * sqdist / (length_scale**2))

    def pairwise(self, X, Y, params):
        """
        Computes pairwise kernel matrix between all points in X and Y
        """
        # X: (N, D), Y: (M, D)
        def k_fn(x):
            return vmap(lambda y: self(x, y, params))(Y)
        return vmap(k_fn)(X)
