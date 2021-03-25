import re, os
from tqdm.auto import tqdm
from jax import lax
import jax.numpy as jnp
from jax import random
import numpy as np
import struct
from jax.experimental import host_callback
from jax.tree_util import tree_leaves

# ===================
# 1. NN specific utils

def one_hot(x, k, dtype=jnp.float32):
    "Create a one-hot encoding of x of size k."
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def read_idx(filename):
    "to open idx file (for the notMNIST dataset)"
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def gen_layer(key, size):
    scale = 1e-2
    subkey1, subkey2 = random.split(key)
    W = scale*random.normal(subkey1, shape=size[0])
    b = scale*random.normal(subkey2, shape=size[1])
    return (W, b)

def init_params(key, M, D, K, L):
    if L <= 1:
        raise ValueError("L must be 2 or more")
    size1 = [(M,D), (M,)]
    sizen = [(M,M), (M,)]
    sizeL = [(K,M), (K,)]
    keys = random.split(key, L)

    list_params = [gen_layer(keys[0], size1)]

    for i in range(1,L-1):
        list_params.append(gen_layer(keys[i], sizen))

    list_params.append(gen_layer(keys[L], sizeL))
    return list_params


# ================
# 2. Generic utils


def set_host_device_count(n):
    """
    From Numpyro
    By default, XLA considers all CPU cores as one device. This utility tells XLA
    that there are `n` host (CPU) devices available to use. As a consequence, this
    allows parallel mapping in JAX :func:`jax.pmap` to work in CPU platform.

    .. note:: This utility only takes effect at the beginning of your program.
        Under the hood, this sets the environment variable
        `XLA_FLAGS=--xla_force_host_platform_device_count=[num_devices]`, where
        `[num_device]` is the desired number of CPU devices `n`.

    .. warning:: Our understanding of the side effects of using the
        `xla_force_host_platform_device_count` flag in XLA is incomplete. If you
        observe some strange phenomenon when using this utility, please let us
        know through our issue or forum page. More information is available in this
        `JAX issue <https://github.com/google/jax/issues/1408>`_.

    :param int n: number of CPU devices to use.
    """
    xla_flags = os.getenv('XLA_FLAGS', '').lstrip('--')
    xla_flags = re.sub(r'xla_force_host_platform_device_count=.+\s', '', xla_flags).split()
    os.environ['XLA_FLAGS'] = ' '.join(['--xla_force_host_platform_device_count={}'.format(n)]
                                       + xla_flags)
    
def wait_until_computed(x):
    for leaf in tree_leaves(x):
        leaf.block_until_ready()


def progress_bar_scan_parallel(num_samples, num_chains):
    """Factory that builds a progress bar decorator along
    with the `set_tqdm_description` and `close_tqdm` functions
    """

    if num_samples > 20:
        print_rate = int(num_samples / 20)
    else:
        print_rate = 1

    remainder = num_samples % print_rate

    tqdm_bars = {}
    finished_chains = []
    for chain in range(num_chains):
        tqdm_bars[chain] = tqdm(range(num_samples), position=chain)
        tqdm_bars[chain].set_description("Compiling.. ", refresh=True,)

    def _update_tqdm(arg, transform, device):
        chain = int(str(device)[4:])
        tqdm_bars[chain].set_description(f"Running chain {chain+1}", refresh=False,)
        tqdm_bars[chain].update(arg)

    def _close_tqdm(arg, transform, device):
        chain = int(str(device)[4:])
        tqdm_bars[chain].update(arg)
        finished_chains.append(chain)
        if len(finished_chains) == num_chains:
            for chain in range(num_chains):
                tqdm_bars[chain].close()

    def _update_progress_bar(iter_num):
        """Updates tqdm progress bar of a JAX loop only if the iteration number is a multiple of the print_rate
        Usage: carry = progress_bar((iter_num, print_rate), carry)
        """

        _ = lax.cond(
            iter_num == 1,
            lambda _: host_callback.id_tap(_update_tqdm, 0, result=iter_num, tap_with_device=True),
            lambda _: iter_num,
            operand=None,
        )
        _ = lax.cond(
            iter_num % print_rate == 0,
            lambda _: host_callback.id_tap(_update_tqdm, print_rate, result=iter_num, tap_with_device=True),
            lambda _: iter_num,
            operand=None,
        )
        _ = lax.cond(
            iter_num == num_samples,
            lambda _: host_callback.id_tap(_close_tqdm, remainder, result=iter_num, tap_with_device=True),
            lambda _: iter_num,
            operand=None,
        )

    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.fori_loop`.
        Note that `body_fun` must be looping over a tuple who's first element is `np.arange(num_samples)`.
        This means that `iter_num` is the current iteration number
        """
        def wrapper_progress_bar(carry, x):
            result = func(carry, x)
            _update_progress_bar(x + 1)
            return result
        return wrapper_progress_bar

    return _progress_bar_scan


# non-parallel version        
def progress_bar_scan(num_samples, message=None):
    "Progress bar for a JAX scan"
    if message is None:
            message = f"Running for {num_samples:,} iterations"
    tqdm_bars = {}

    if num_samples > 20:
        print_rate = int(num_samples / 20)
    else:
        print_rate = 1
    remainder = num_samples % print_rate

    def _define_tqdm(arg, transform):
        tqdm_bars[0] = tqdm(range(num_samples))
        tqdm_bars[0].set_description(message, refresh=False)

    def _update_tqdm(arg, transform):
        tqdm_bars[0].update(arg)

    def _update_progress_bar(iter_num):
        "Updates tqdm progress bar of a JAX scan or loop"
        _ = lax.cond(
            iter_num == 0,
            lambda _: host_callback.id_tap(_define_tqdm, None, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            (iter_num % print_rate == 0) & (iter_num != num_samples-remainder),
            lambda _: host_callback.id_tap(_update_tqdm, print_rate, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = lax.cond(
            # update tqdm by `remainder`
            iter_num == num_samples-remainder,
            lambda _: host_callback.id_tap(_update_tqdm, remainder, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

    def _close_tqdm(arg, transform):
        tqdm_bars[0].close()

    def close_tqdm(result, iter_num):
        return lax.cond(
            iter_num == num_samples-1,
            lambda _: host_callback.id_tap(_close_tqdm, None, result=result),
            lambda _: result,
            operand=None,
        )


    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(num_samples)`,
        or be looping over a tuple who's first element is `np.arange(num_samples)`
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x
            _update_progress_bar(iter_num)
            result = func(carry, x)
            return close_tqdm(result, iter_num)

        return wrapper_progress_bar

    return _progress_bar_scan
