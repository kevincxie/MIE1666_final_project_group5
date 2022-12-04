from typing import NamedTuple, Callable, Any
from dataclasses import dataclass

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array

import optax
import jaxopt

def problem_dataloader(problems, batch_size):
    n_problems = problems[0].shape[0] # HACK
    n_batches_per_epoch = math.ceil(n_problems / batch_size)
    for batch_i in range(n_batches_per_epoch):
        if batch_i == n_batches_per_epoch-1:
            batch_probp = jax.tree_map(lambda x: x[batch_i*batch_size:], problems)
        else:
            batch_probp = jax.tree_map(lambda x: x[batch_i*batch_size: (batch_i+1)* batch_size], problems)
        yield batch_probp

def mock(step_size, iterations, **kwargs):
    """ Identity operator doesn't improve solutions at all """
    def improver(key, iterator, f, x):
        del key, iterator
        return x, None
    return improver

def sgld(step_size: float, iterations: int, eta: float=0.01,
         gamma: float=0.55, **kwargs) -> Callable:
    """ Applies sgld to sample from exp(-tau*L) """
    def improver(key, iterator, f, x):
        # TODO(Y) : try to find a way to reuse state
        seed = (jax.random.uniform(key) * 1000).astype(int)
        optim = optax.noisy_sgd(step_size, eta, gamma, seed)
        state = optim.init(x)

        cost_fn = jax.vmap(jax.value_and_grad(f))
        #grad = jax.vmap(jax.vmap(jax.grad(f), in_axes=0, out_axes=0), in_axes=0, out_axes=0)

        def update(i, paramstates):
            x, state = paramstates
            c_i, dx_i = cost_fn(x)
            updates, state = optim.update(dx_i, state)
            x_i = optax.apply_updates(x, updates)
            return (x_i, state)

        x, state = jax.lax.fori_loop(0, iterations, update, (x, state))
        c, _ = cost_fn(x)

        return x, c, state
    return improver

