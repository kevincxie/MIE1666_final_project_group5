
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float32
from typing import Tuple

import numpy as onp

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ProblemParamType = Tuple[Float32[jnp.ndarray, "batch patch state"], Float32[jnp.ndarray, "batch patch"]]
TrajectoryType = Float32[Array, "batch class length state"]

PHI_STATE_DIM = 2 # Size of the problem desc per dimension

def plot_background(fig, ax, psi : ProblemParamType):
    loc, rad = psi
    for l, r in zip(loc, rad):
        circle = Circle(l, r)
        ax.add_patch(circle)
        
def plot_solution(fig, ax, solution : Float32[Array, "length state"], linestyle='--'):
    xs = solution[:, 0]
    ys = solution[:, 1]
    ax.plot(xs, ys)

def plot_single_problem(fig, ax, psi : ProblemParamType, soln : TrajectoryType, modes : int):
    plot_background(fig, ax, psi)
    print(soln, modes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    soln = soln.reshape(soln.shape[0], soln.shape[1], -1, PHI_STATE_DIM)

    for i in range(modes):
        plot_solution(fig, ax, soln[0, i])

    #ax.tick_params(which='both', bottom=False, top=False, labelbottom=False, labelleft=False)
    return fig, ax

def make_problem(patches : int = 2, box : Float32[jnp.ndarray, "2 state"] = jnp.array([[0., 0.], [1., 1.]]), min_rad : float=0.01, max_rad: float=0.05):
    def cost(q : TrajectoryType, psi : ProblemParamType) -> Float32[Array, "batch"]:
        loc, rad = psi
        loc = loc[:, None, None, :, :]
        rad = rad[:, None, None, :]
        qb = q[:, :, :, None, :]
        
        d = jnp.linalg.norm(qb - loc, axis=-1)
        signed_dist_cost = (d - rad).sum(axis=-1).min(axis=-1)
        path_length = jnp.linalg.norm(q[:, :, 1:, :] - q[:, :, :-1, :], axis=-1)
        path_length_costs = path_length.sum(axis=-1)
        return + signed_dist_cost
        
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
        order = jnp.argsort(rad, axis=-1)
        loc = jnp.take_along_axis(loc, order[:, :, None], axis=1)
        rad = jnp.take_along_axis(rad, order, axis=1)
        
        sample_d = random.normal(key, shape=loc.shape)
        sample_d /= jnp.linalg.norm(sample_d, axis=-1, keepdims=True)
        q = loc #+ rad[:, :, None] * sample_d
        
        return q
        
    def get_phi(psi : ProblemParamType) -> Float32[jnp.ndarray, "batch patch state"]:
        return psi[0]
        
    return sample_problem_params, get_phi, cost, solver
    

if __name__ == '__main__':
    
    get_params, get_phi, cost, solver = make_problem(patches=4)
    key = random.PRNGKey(0)
    
    params = get_params(key, 1)
    loc, rad = params
    
    phi = get_phi(params)
    solution = solver(key, params)[0]
    
    fig, ax = plt.subplots()
    
        
    fig.savefig("gcs_visuals.png")
        
    
        
    

    
    
    
