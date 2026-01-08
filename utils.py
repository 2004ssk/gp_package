# utils.py

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

def generate_synthetic_data(n=20):
    """
    Generate synthetic data from a known function with noise
    """
    key = jax.random.PRNGKey(42)
    X = jnp.linspace(-3, 3, n).reshape(-1, 1)
    noise = 0.1 * jax.random.normal(key, shape=(n,))
    Y = jnp.sin(X[:, 0]) + noise
    return X, Y

def gp_plot(X_train, y_train, X_test, pred_mean, pred_cov, title="GP Prediction"):
    """
    - Training data as red crosses.
    - Predicted mean as a blue line.
    - Confidence interval (mean ± 2 standard deviations) as a shaded area
    """
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    X_test_np = np.array(X_test)
    pred_mean_np = np.array(pred_mean)
    pred_cov_np = np.array(pred_cov)
    pred_std_np = np.sqrt(np.diag(pred_cov_np))

    plt.figure(figsize=(10, 6))
    plt.plot(X_test_np, pred_mean_np, 'b-', label="Predicted Mean")
    plt.fill_between(
        X_test_np.flatten(),
        pred_mean_np - 2 * pred_std_np,
        pred_mean_np + 2 * pred_std_np,
        color="blue",
        alpha=0.2,
        label="Confidence Interval (2 std)",
    )
    plt.plot(X_train_np, y_train_np, "rx", label="Training Data")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.show()
