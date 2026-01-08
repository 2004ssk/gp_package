# gp.py

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
from .kernels import BaseKernel
from .means import BaseMean

class GPParameters:
    """
    Class that stores Gaussian Process hyperparameters
    """
    def __init__(self, length_scale: float, variance: float, noise: float):
        self.length_scale = length_scale
        self.variance = variance
        self.noise = noise

    def to_dict(self):
        return {
            "length_scale": self.length_scale,
            "variance": self.variance,
            "noise": self.noise,
        }


class GaussianProcess:
    """
    A Gaussian Process model with kernel and mean function
    """

    def __init__(self, kernel: BaseKernel, mean: BaseMean):
        self.kernel = kernel
        self.mean = mean

    def marginal_log_likelihood(self, X, y, params):
        """
        Computes negative log marginal likelihood

        Parameters:
          - X: (N, D) training inputs
          - y: (N,) training targets
          - params: a dictionary containing GP parameters

        Returns:
          - Negative log marginal likelihood.
        """
        m = self.mean(X, params)
        y_centered = y - m

        K = self.kernel.pairwise(X, X, params)
        K = K + jnp.eye(X.shape[0]) * params["noise"]

        L = cho_factor(K)
        alpha = cho_solve(L, y_centered)

        data_fit = 0.5 * jnp.dot(y_centered, alpha)
        complexity = jnp.sum(jnp.log(jnp.diag(L[0])))
        constant = 0.5 * X.shape[0] * jnp.log(2.0 * jnp.pi)
        log_likelihood = data_fit + complexity + constant
        return -log_likelihood

    def predict(self, X_train, y_train, X_test, params):
        """
        Computes predicted mean and covariance at test inputs X_test

        Parameters:
          - X_train: training inputs
          - y_train: training targets
          - X_test: test inputs
          - params: a dictionary containing GP parameters

        Returns:
          - pred_mean: the predictive mean
          - pred_cov: the predictive covariance matrix
        """
        m_train = self.mean(X_train, params)
        K_train = self.kernel.pairwise(X_train, X_train, params)
        K_train += jnp.eye(X_train.shape[0]) * params["noise"]

        m_test = self.mean(X_test, params)
        K_cross = self.kernel.pairwise(X_train, X_test, params)
        K_test = self.kernel.pairwise(X_test, X_test, params)

        L = cho_factor(K_train)
        alpha = cho_solve(L, y_train - m_train)

        pred_mean = m_test + jnp.dot(K_cross.T, alpha)

        v = cho_solve(L, K_cross)
        pred_cov = K_test - jnp.dot(K_cross.T, v)

        return pred_mean, pred_cov
