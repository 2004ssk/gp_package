# gp_package

A lightweight, minimal-dependency Gaussian process library built in JAX. Written from scratch as part of research at Cornell's Bindel Lab.

The goal was a clean, hackable GP implementation that avoids the overhead of GPyTorch or GPflow — useful when you want full control over the model, the kernel, or the training loop without fighting a large framework.

---

## Features

- Gaussian process regression with customizable kernels and mean functions
- Kernels implemented from scratch in JAX (see `kernels.py`)
- Pluggable mean functions (see `means.py`)
- Training utilities with gradient-based marginal likelihood optimization (see `train.py`)
- JIT-compiled and fully differentiable — works with JAX's `grad`, `jit`, and `vmap`
- No dependency on GPyTorch, GPflow, or any other GP framework

---

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/2004ssk/gp_package.git
cd gp_package
pip install -r requires.txt
```

---

## Quickstart

```python
import jax.numpy as jnp
from gp import GaussianProcess
from kernels import RBFKernel
from means import ZeroMean

# Training data
X_train = jnp.linspace(0, 5, 20).reshape(-1, 1)
y_train = jnp.sin(X_train).squeeze() + 0.1 * jax.random.normal(key, shape=(20,))

# Define model
kernel = RBFKernel(lengthscale=1.0, variance=1.0)
mean = ZeroMean()
gp = GaussianProcess(kernel=kernel, mean=mean, noise=0.01)

# Fit and predict
gp.fit(X_train, y_train)

X_test = jnp.linspace(0, 5, 100).reshape(-1, 1)
mean_pred, var_pred = gp.predict(X_test)
```

---

## Repo Structure

```
gp_package/
├── gp.py          # Core GaussianProcess class — fit, predict, log marginal likelihood
├── kernels.py     # Kernel functions (RBF, Matérn, etc.)
├── means.py       # Mean functions (zero mean, constant, linear)
├── train.py       # Hyperparameter optimization via gradient descent on log marginal likelihood
├── utils.py       # Numerical utilities (Cholesky solves, jitter, etc.)
├── __init__.py
└── requires.txt
```

---

## Design Notes

**Why JAX?** JAX's `grad` and `jit` make differentiating through the log marginal likelihood and JIT-compiling the full training loop straightforward. No manual gradient derivations needed, and the resulting code stays close to the math.

**Numerical stability** — covariance matrices are regularized with a small jitter term before Cholesky decomposition. All linear solves go through Cholesky rather than direct inversion.

**Minimal dependencies** — the only hard requirements are JAX and NumPy. No framework lock-in.

---

## Requirements

- `jax`
- `jaxlib`
- `numpy`

See `requires.txt` for pinned versions.

---

## Background

Built during a research assistantship at the [Bindel Lab](https://www.cs.cornell.edu/~bindel/) at Cornell University. The package was developed alongside work on Bayesian optimization and parameter-free SGD algorithms for convex objectives.

---

## License

MIT
