# init.py

"""
gaussian_process: A minimal Gaussian Process library using JAX
"""

__version__ = "0.1.0"

from .kernels import RBFKernel
from .means import ZeroMean
from .gp import GaussianProcess, GPParameters  
from .train import train_gp
from .utils import generate_synthetic_data, gp_plot  
