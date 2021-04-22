################################################################
########## INVERSE AUTOREGRESSIVE (NORMALIZING) FLOWS ##########
################################################################

import jax.numpy as jnp
import jax.random as rand
from jax import vmap
from MaskedAutoReg import DenseAutoReg, variance_scaling

def AutoRegFlow (z, mu_param, log_sd_param, mu_act, log_sd_act):
    """
    Autoregressive flow
    z: input vector
    mu_param: pytree of the form [([columns of W x M], b), ...] 
    mu_act: either [jnn.act(), ...] same length as _param or [jnn.act()]
    """
    x = jnp.zeros_like(z)
    for _ in range(z.shape[0]):
        mu = DenseAutoReg(x, mu_param, mu_act)
        log_sd = DenseAutoReg(z, log_sd_param, log_sd_act)
        x = jnp.exp(log_sd)*z + mu
    
    return x
    
def InvAutoRegFlow (z, mu_param, log_sd_param, mu_act, log_sd_act):
    """
    Inverse autoregressive flow Kingma et al 2016.
    z: input vector
    _param: pytree of the form [([columns of W x M], b), ...] 
    _act: either [jnn.act(), ...] same length as _param or [jnn.act()]
    """
    mu = DenseAutoReg(z, mu_param, mu_act)
    log_sd = DenseAutoReg(z, log_sd_param, log_sd_act)
    
    return (z - mu)/jnp.exp(log_sd)
    # return jnp.exp(log_sd)*z + mu
    
def MakeFlow (z, parameters, activations, invert = True):
    """
    Repeat Inverse autoregressive flow on input.
    z: input vector
    parameters: pytree of the form [(mu_param, log_sd_param), ...]
        each tuple a repetition of the flow
    activations: pytree of the form [(mu_act, log_sd_param), ...]
    invert: invert the order of inputs at each repetition
    """
    log_det_jac = 0.
    for param, act in zip(parameters, activations):
        log_det_jac -= jnp.sum(DenseAutoReg(z, param[1], act[1]))
        # log_det_jac += jnp.sum(DenseAutoReg(z, param[1], act[1]))
        z = InvAutoRegFlow(z, *param, *act)
        if invert:
            z = z[::-1]
    if invert and len(parameters) % 2 > 0:
        z = z[::-1]
    
    return z, log_det_jac

def MCKLDiv (Z, parameters, activations, log_target, invert = True):
    """
    Monte Carlo estimate of (reverse) Kullback-Leiber divergence.
    Z: jnp.array with observations of q(z) as rows
    paramters: pytree of the form [(mu_param, log_sd_param), ...]
    activations: pytree of the form [(mu_act, log_sd_param), ...]
    log_target: logarithm of target distribution p(x) to approach with flow
    invert: invert the order of inputs at each repetition
    """
    X, LDJ = vmap(MakeFlow, (0, None, None, None), 0)(Z, parameters, activations, invert)
    KL = -jnp.sum( vmap(log_target, 0, 0)(X) + LDJ )
    
    return KL/(jnp.shape(Z)[0])
    
def init_rand_param (d, K, hidden_layers, seed = 123, rng = variance_scaling(1., "truncated_normal")):
    """
    Initializes random parameters for K Inverse autoregressive flow.
    d: dimension of input (z)
    K: number of repetitions of the flow
    hidden_layers: pytree of the form [(# hidden layers mu, # hidden layers log sd), ...]
        each tuple a repetition of the flow
    seed: seed PRNG
    rng: RNG
    """
    assert K == len(hidden_layers)
    keys = [rand.PRNGKey(seed)]
    parameters = []
    for f in range(K):
        keys = rand.split(keys[0], 1+d+1)
        mu_layers = [([rng(keys[i+1], d/1., (d-(i+1), )) for i in range(d)], rng(keys[d+1], d/1., (d, )))]
        keys = rand.split(keys[0], 1+d+1)
        log_sd_layers = [([rng(keys[i+1], d/1., (d-(i+1), )) for i in range(d)], rng(keys[d+1], d/1., (d, )))]
        for h in range(hidden_layers[f][0]):
            keys = rand.split(keys[0], 1+d+1)
            mu_layers.append(([rng(keys[i+1], d/1., (d-i, )) for i in range(d)], rng(keys[d+1], d/1., (d, ))))
        for h in range(hidden_layers[f][1]):
            keys = rand.split(keys[0], 1+d+1)
            log_sd_layers.append(([rng(keys[i+1], d/1., (d-i, )) for i in range(d)], rng(keys[d+1], d/1., (d, ))))
        parameters.append((mu_layers, log_sd_layers))
        
    return parameters
