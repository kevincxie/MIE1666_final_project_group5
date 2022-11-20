
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float32
from typing import Tuple

import numpy as onp

ProblemParamType = Tuple[Float32[jnp.ndarray, "batch patch state"], Float32[jnp.ndarray, "batch patch"]]
TrajectoryType = Tuple[Float32[Array, "batch length state"]]

def make_problem(patches : int = 2, box : Float32[jnp.ndarray, "2 state"] = jnp.array([[0., 0.], [1., 1.]]), min_rad : float=0.01, max_rad: float=0.05):
    def cost(psi : ProblemParamType, q : TrajectoryType) -> Float32[Array, "batch"]:
        loc, rad = psi
        d = jnp.norm(q - loc, axis=-1)
        signed_dist_cost = (d - rad).sum(axis=-1)
        path_length = jnp.norm(q[:, 1:, :] - q[:, :-1, :], axis=-1)
        path_length_costs = path_length.sum(axis=-1)
        return path_length_costs + signed_dist_cost
        
    def sample_problem_params(key : random.PRNGKey, batch_size : int) -> ProblemParamType:
        # Add padding
        padded_min = box[0] + max_rad
        padded_max = box[1] - max_rad
        
        loc_k, rad_k = random.split(key, 2)
        
        loc = random.uniform(key, shape=(batch_size, patches, box.shape[1]), minval=padded_min, maxval=padded_max)
        rad = random.uniform(key, shape=(batch_size, patches), minval=min_rad, maxval=max_rad)
        
        return loc, rad
        
    def solver(key : random.PRNGKey, problem_params : ProblemParamType) -> TrajectoryType:
        loc, rad = problem_params
        sample_d = random.normal(key, shape=loc.shape)
        sample_d /= jnp.linalg.norm(sample_d, axis=-1, keepdims=True)
        
        q = loc + rad[:, :, None] * sample_d
        return q
        
    def get_phi(psi : ProblemParamType) -> Float32[jnp.ndarray, "batch patch state"]:
        return psi[0]
        
    return sample_problem_params, get_phi, cost, solver
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    
    get_params, get_phi, cost, solver = make_problem(patches=4)
    key = random.PRNGKey(0)
    
    params = get_params(key, 1)
    loc, rad = params
    
    phi = get_phi(params)
    solution = solver(key, params)[0]
    
    fig, ax = plt.subplots()
    
    for l, r in zip(loc[0], rad[0]):
        circle = Circle(l, r)
        ax.add_patch(circle)
        
    xs = solution[:, 0]
    ys = solution[:, 1]
    print(xs, ys)
    ax.plot(xs, ys)
    fig.savefig("gcs_visuals.png")
        
    
        
    

    
    
    
