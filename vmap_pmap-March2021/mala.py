import numpy as onp

import jax.numpy as jnp
from jax import random, partial, jit, lax
import time

from util import progress_bar_scan_parallel


@partial(jit, static_argnums=(3,6))
def mala_kernel(key, paramCurrent, paramGradCurrent, log_post, logpostCurrent, dt, dim):
    subkey1, subkey2 = random.split(key)
    paramProp = paramCurrent + dt*paramGradCurrent + jnp.sqrt(2*dt)*random.normal(key=subkey1, shape=(dim,))
    new_log_post, new_grad = log_post(paramProp)

    term1 = paramProp - paramCurrent - dt*paramGradCurrent
    term2 = paramCurrent - paramProp - dt*new_grad
    q_new = -0.25*(1/dt)*jnp.dot(term1, term1)
    q_current = -0.25*(1/dt)*jnp.dot(term2, term2)

    log_ratio = new_log_post - logpostCurrent + q_current - q_new
    acceptBool = jnp.log(random.uniform(key=subkey2)) < log_ratio
    paramCurrent = jnp.where(acceptBool, paramProp, paramCurrent)
    current_grad = jnp.where(acceptBool, new_grad, paramGradCurrent)
    current_log_post = jnp.where(acceptBool, new_log_post, logpostCurrent)
    accepts_add = jnp.where(acceptBool, 1,0)
    return paramCurrent, current_grad, current_log_post, accepts_add


def mala_sampler(key, num_samples, dt, val_and_grad_log_post, x_0, num_chains=1):
    dim, = x_0.shape
    
    @progress_bar_scan_parallel(num_samples, num_chains)
    def mala_step(carry, x):
        key, paramCurrent, gradCurrent, logpostCurrent, accepts = carry
        key, subkey = random.split(key)
        paramCurrent, gradCurrent, logpostCurrent, accepts_add = mala_kernel(subkey, paramCurrent, gradCurrent, val_and_grad_log_post, logpostCurrent, dt, dim)
        accepts += accepts_add
        return (key, paramCurrent, gradCurrent, logpostCurrent, accepts), (paramCurrent, gradCurrent)

    paramCurrent = x_0
    logpostCurrent, gradCurrent = val_and_grad_log_post(x_0)
    carry = (key, paramCurrent, gradCurrent, logpostCurrent, 0)
    (_, _, _, _, accepts), (samples, grads) = lax.scan(mala_step, carry, jnp.arange(num_samples))
    return samples, grads, 100*(accepts/num_samples)
