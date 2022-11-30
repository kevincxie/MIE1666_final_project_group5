from typing import NamedTuple, Callable, Any
from dataclasses import dataclass


import jax
import jax.numpy as jnp
from jaxtyping import Array


import optax
import jaxopt


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
    