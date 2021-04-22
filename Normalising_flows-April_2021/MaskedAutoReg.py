################################################################
########### MASKED (BATCH) AUTOREGRESSIVE CONDITIONER ##########
################################################################
# and other helper functions for normalizing flows

import jax.numpy as jnp
import jax.random as rand

def DenseAutoReg (z, parameters, activations):
    """
    (Masked) Autoregressive neural network (square) Germain et al 2015.
    z: input vector
    parameters: pytree of the form [([columns of W x M], b), ...] 
        each tuple a layer [initial, hidden1, ...]
        the columns ignore 0 values, hence decreasing
        first W x M matrix has 0 in diag, hence empty list as last column (exclusive)
        columns of W x M and b are jnp.arrays
    activations: either [jnn.act(), ...] same length as parameters or [jnn.act()]
    """ 
    if len(activations) == 1 and len(parameters) > 1:
        h = len(parameters)
        activations = [activations[0] for _ in range(h-1)] + [None]
    else:
        assert len(parameters) == len(activations)
    for j, (W, b) in enumerate(parameters):
        if j == 0:
            W = [jnp.concatenate([jnp.zeros(i+1), w]) for i, w in enumerate(W)]
        else:
            W = [jnp.concatenate([jnp.zeros(i), w]) for i, w in enumerate(W)]
        W = jnp.array(W)
        z = jnp.dot(z, W) + b
        if activations[j]:
            z = activations[j](z)
    return z

def variance_scaling(scale, distribution, dtype=jnp.float32):
  def init(key, denominator, shape, dtype=dtype):
    variance = jnp.array(scale / denominator, dtype=dtype)
    if distribution == "truncated_normal":
      # constant is stddev of standard normal truncated to (-2, 2)
      stddev = jnp.sqrt(variance) / jnp.array(.87962566103423978, dtype)
      return rand.truncated_normal(key, -2, 2, shape, dtype) * stddev
    elif distribution == "normal":
      return rand.normal(key, shape, dtype) * jnp.sqrt(variance)
    elif distribution == "uniform":
      return rand.uniform(key, shape, dtype, -1) * jnp.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")
  return init