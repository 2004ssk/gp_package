# train.py

import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.example_libraries import optimizers

def train_gp(gp_model, X, y, init_params, num_steps=100, learning_rate=0.01):
    """
    Train GP model hyperparameters by minimizing the negative log marginal likelihood
    
    Parameters:
      - gp_model: a GaussianProcess object
      - X: training inputs
      - y: training targets
      - init_params: initial hyperparameters 
      - num_steps: number of optimization steps
      - learning_rate: step size for the Adam optimizer
      
    Returns:
      - final_params: the optimized hyperparameters 
    """
    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    opt_state = opt_init(init_params)

    def loss_fn(params):
        return gp_model.marginal_log_likelihood(X, y, params)

    @jit 
    def step(i, opt_state):
        params = get_params(opt_state)
        grads = grad(loss_fn)(params)
        return opt_update(i, grads, opt_state)

    for i in range(num_steps):
        opt_state = step(i, opt_state)

    return get_params(opt_state)
