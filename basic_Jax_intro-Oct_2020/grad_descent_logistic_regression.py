import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from linetimer import CodeTimer

# I use the {-1, 1} encoding.
# I find that using `inner` means you don't need `vmap`.
# This might not apply in higher dimensions


def add_ones(x):
    n, _ = np.shape(x)
    return np.append(np.ones((n, 1)), x, axis=-1)


@jax.jit
def logit(p):
    return np.log(p) - np.log(1. - p)


@jax.jit
def expit(x):
    return 1./(1. + np.exp(-x))


@jax.jit
def expitpm1(x):
    return 2./(1. + np.exp(-x)) - 1.


def sim_data(n, p, sigma, key):
    assert(n > 0)
    assert(p > 0)
    beta_key, covar_key, data_key, key = jax.random.split(key, 4)
    beta = sigma * jax.random.normal(beta_key, (p,))
    x = add_ones(jax.random.normal(covar_key, (n, p-1)))
    probs = expit(np.inner(x, beta))
    y = 2 * jax.random.bernoulli(data_key, probs) - 1
    return y, x, beta, key


# numerically safer
@jax.jit
def partial_loglike(beta, x, y):
    return (y - 1)/2 * np.inner(beta, x) - np.log(1.0 + np.exp(-np.inner(beta, x)))


@jax.jit
def batch_loglike(beta, x, y):
    return jax.vmap(partial_loglike, (None, 0, 0))(beta, x, y)


@jax.jit
def loglike(beta, x, y):
    return np.sum(partial_loglike(beta, x, y))


@jax.jit
def log_prior(beta, sigma):
    return -0.5 * np.inner(beta, beta)/sigma**2


@jax.jit
def log_post(beta, x, y, sigma):
    return log_prior(beta, sigma) + loglike(beta, x, y)


@jax.jit
def batched_log_post(beta, x, y, sigma):
    return log_prior(beta, sigma) + np.sum(batch_loglike(beta, x, y))


def grad_descent(logp, beta, x, y, sigma, learning_rate=0.01, maxit=1000):
    trace_log_like = []
    for it in range(maxit):
        l, g = jax.value_and_grad(logp)(beta, x, y, sigma)
        beta = beta + learning_rate * g
        trace_log_like.append(l)
    return beta, trace_log_like


def RMSE(x, y):
    return np.sqrt(np.mean((x - y)**2))


ndata = 10000
ncovar = 15
key = jax.random.PRNGKey(0)
sig = 0.1
y, x, true_beta, key = sim_data(ndata, ncovar, 0.1, key)

if ncovar == 2:
    grid_x = add_ones(np.linspace(-3, 3, 1000).reshape(-1, 1))
    plt.scatter(x[:, 1], y)
    plt.plot(grid_x[:, 1], expitpm1(np.inner(grid_x, true_beta)))
    plt.show()

with CodeTimer("naive log_pos"):
    jax.value_and_grad(log_post)(true_beta, x, y, sig)

with CodeTimer("batch log_pos"):
    jax.value_and_grad(batched_log_post)(true_beta, x, y, sig)

with CodeTimer("naive"):
    beta = sig * jax.random.normal(key, (ncovar,))
    hat_beta, ll_trace = grad_descent(log_post, beta, x, y, sig, 0.00025, 500)


true_beta
hat_beta
RMSE(true_beta, hat_beta)
plt.plot(ll_trace[::1])
plt.show()


if ncovar == 2:
    plt.scatter(x[:, 1], y)
    plt.plot(grid_x[:, 1], expitpm1(np.inner(grid_x, true_beta)))
    plt.plot(grid_x[:, 1], expitpm1(np.inner(grid_x, hat_beta)))
    plt.show()
