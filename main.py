import os
import importlib

import jax
from jax import random
import jax.numpy as jnp
import optax
import equinox as eqx
import equinox.nn as nn
import pdb

import matplotlib.pyplot as plt
import seaborn as sns

from argparse import ArgumentParser

from typing import Callable
from functools import partial

import problems
import svgd_utils



class ZDecoder(eqx.Module):
    weight: jnp.ndarray
    weight2: jnp.ndarray
    #bias: jnp.ndarray
    region_params: jnp.ndarray
    
    model: nn.MLP

    z_size: int
    levels: int
    regions: int

    def __init__(self, levels, regions, latent_dim, phi_size, out_size, key):
        assert(levels <= latent_dim and latent_dim % levels == 0)

        self.z_size = latent_dim // levels
        self.levels = levels
        self.regions = regions

        wkey, bkey, Zkey = jax.random.split(key, num=3)
    #    self.weight1 = jax.random.normal(wkey, (out_size, latent_dim))
        self.weight2 = jax.random.normal(bkey, (out_size, phi_size))
        self.weight = jnp.ones((out_size, in_size))
        #self.bias = jax.random.normal(bkey, (out_size,))
        self.region_params = jax.random.normal(Zkey, (levels, regions, self.z_size))
        
        self.model = nn.MLP(in_size, out_size, 64, 2, key=wkey)
 
    @partial(jax.vmap, in_axes=(None, 0), out_axes=0)   
    @partial(jax.vmap, in_axes=(None, 0), out_axes=0)   
    def eval_mlp(self, x):
        return self.model(x)

    def __call__(self, phi):
        phi = phi.reshape(phi.shape[0], -1) # Treat all the enviorenment parameters as one long vector..
        batch_size, dim = phi.shape

        regions, levels, z_size = self.regions, self.levels, self.z_size

        i = jnp.mgrid[(slice(0, regions, 1),)*levels]
        V_alphas = self.region_params[jnp.arange(levels), i.T].reshape(-1, levels*z_size)
      #  jax.debug.print("V_alpha: {}", V_alphas)

        phi = jnp.broadcast_to(phi.reshape(batch_size, 1, dim), (batch_size, V_alphas.shape[0], dim))
        V_alphas = jnp.broadcast_to(V_alphas, (batch_size,)+V_alphas.shape)

        X = jnp.concatenate([V_alphas, phi], axis=-1) # batch, nregions*nlevels, z_size+phi_dim
        #qs = V_alphas + jnp.matmul(phi, self.weight2.T)
      #  jax.debug.print("{}", phi)
        #qs = jnp.matmul(X, self.weight.T) #[batch, nregions*nlevels, q_size]
        
        # print(X.shape)
        qs = self.eval_mlp(X)
        #qs = V_alphas
#        jax.debug.print("{}", V_alphas)

        return qs


# def create_dataset(num_data, nwalls=2, batchsize=3):
#     phi = []
#     q_star = []
#     samp_prob, get_phi, cost, mock_sol = get_toy_problem_functions(nwalls)

#     for i in range (num_data):
#         key = jax.random.PRNGKey(i)
#         probp = samp_prob(key, batchsize)
#         q_star.append(mock_sol(probp)[0])
#         phi.append(get_phi(probp))
#     return phi, q_star


def eval(model, psi, cost, get_phi, state_dim):
    phi = get_phi(psi)
    batch_size = phi.shape[0]

    qh = model(phi)
    qh = qh.reshape(batch_size, qh.shape[1], -1, state_dim)
    c = cost(qh, psi)
    best_qs = c.argmin(axis=1)
    best_q = qh[jnp.arange(batch_size), best_qs]
    return best_q

def train(args, model, key):
    samp_prob, get_phi, cost, mock_sol = args.problem_inst.make_problem(args.prob_dim)
    phi_size = args.prob_dim
    in_size, out_size = args.latent_dim + phi_size, args.prob_dim

    @eqx.filter_value_and_grad
    def compute_loss(model, phi, q_star):
        qs = model(phi)

        q_fit = (qs - q_star[:, None, :])**2
        q_fit = jnp.sum(q_fit, axis=-1)
      #  jax.debug.print("q_fit{}, phi:{}", q_fit, phi)

        # inner minimization
        best_qs = jnp.min(q_fit, axis=1)
        # expectation
        loss = jnp.mean(best_qs, axis=0)
        return loss

    @eqx.filter_value_and_grad
    def compute_loss_q(qs, q_star):
        q_fit = (qs - q_star[:, None, :])**2
        q_fit = jnp.sum(q_fit, axis=-1)
        # inner minimization
        best_qs = jnp.min(q_fit, axis=1)
        # expectation
        loss = jnp.mean(best_qs, axis=0)
        return loss

    # Important for efficiency whenever you use JAX: wrap everything into a single JIT
    # region.
    @eqx.filter_jit
    def make_step(model, phi, q_star, opt_state):
        loss, grads = compute_loss(model, phi, q_star)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    @ eqx.filter_jit
    def make_step_sgld(qs, q_star, opt_state):
        loss, grads = compute_loss_q(qs, q_star)
        updates, opt_state = q_optim.update(grads, opt_state)
        qs = eqx.apply_updates(qs, updates)
        return qs, loss, opt_state

    optim = optax.adam(optax.exponential_decay(args.lr, 10, args.decay))
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    for epoch in range(args.epochs):
        #for sample in range(q_stars_mock.shape[0]):
        _, key_sample, key_solve = jax.random.split(key, 3)
       # key_sample = key_solve = key
        probp = samp_prob(key_sample, batch_size=args.problem_batch_size)
        phi = get_phi(probp).reshape(args.problem_batch_size, -1)

        if args.sgld: # SGLD
            q_star = mock_sol(key_solve, probp)
            q_star = q_star.reshape(args.problem_batch_size, -1)
            qs = model(phi)
            q_optim = optax.noisy_sgd(args.lr, eta=0.005, gamma=0.85)
            q_opt_state = q_optim.init(qs)
            for iter in range(500):
                qs, loss, q_opt_state = make_step_sgld(qs, q_star, q_opt_state)
                if (iter + 1) % 100 == 0:
                    print(f"SGLD epoch={iter+1}, loss={loss : .4f}")
                
        else: # SVGD
            q_star = mock_sol(key_solve, probp)
            q_star = q_star.reshape(args.problem_batch_size, -1)
            svgd = svgd_utils.SVGD(model(phi), args.num_particles, args.seed)
            q_star = svgd.optimize(epochs=1000, gt=q_star, svgd_r=1)

        loss, model, opt_state = make_step(model, phi, q_star, opt_state)

        loss = loss.item()
        #losses.append(loss)
        if (epoch + 1) % 100 == 0:
            print(f"epoch={epoch+1}, loss={loss : .4f}")
            

    return model


def plot_solutions(args, psi, gt, qs, path):
    sns.set_style('whitegrid')
    batches = qs.shape[0]
    fig, axes = plt.subplots(args.rows, batches // args.rows, figsize=(args.plot_width, args.plot_height))
    if batches // args.rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, ax in enumerate(axes):
        phi = (psi[0][i], psi[1][i])
        q = qs[i]
        gt_ = gt[i]
        args.problem_inst.plot_single_problem(fig, ax, phi, q[None, :], q.shape[0])
  #      args.problem_inst.plot_single_problem(fig, ax, phi, gt_.reshape(1, 1, -1), 1)

    fig.savefig(os.path.join(path, "plots.png"))

def test(args, model, key):
    samp_prob, get_phi, cost, mock_sol = args.problem_inst.make_problem(args.prob_dim)
    key_sample, key_solve = jax.random.split(key)
    #key_sample = key_solve = key
    psi = samp_prob(key_sample, args.test_batch_size)
    gt = mock_sol(key_solve, psi)
    phi = get_phi(psi)
    qs = model(phi)

    best_q = eval(model, psi, cost, get_phi, args.problem_inst.PHI_STATE_DIM)
    err = jnp.linalg.norm(gt - best_q, axis=-1)

    mean_error = jnp.mean(err)
    std_dev = jnp.std(err)

    if args.plot:
        plot_solutions(args, psi, gt, qs, args.results_path)

    return mean_error, std_dev


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--problem", type=str, default="maze_1d")

    parser.add_argument("--problem_batch_size", type=int, default=50,
            help="For each iteration how many problems to sample")
    parser.add_argument("--epochs", type=int, default=100,
            help="Total iteration count")

    parser.add_argument("--levels", type=int, default=2,
            help="Number of levels to the problem,"
            + "can't be greater than latent dimension")
    parser.add_argument("--regions", type=int, default=2,
            help="Number of voronoi regions per level")

    parser.add_argument("--test_batch_size", type=int, default=4,
            help="For testing how many problems to sample")
    parser.add_argument("--latent_dim", type=int, default=2,
            help="Dimensions in the latent space")
    parser.add_argument("--prob_dim", type=int, default=2,
            help="Number of dimensions in the problem," +
            "corresponds to the number of walls")

    parser.add_argument("--plot", action='store_true', default=False)
    parser.add_argument("--rows", type=int, default=1, help="Number of columns in the plot grid")
    parser.add_argument("--plot_height", type=int, default=6)
    parser.add_argument("--plot_width", type=int, default=8)

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--decay", type=float, default=1.)

    parser.add_argument("--results_path", type=str, default=".")
    parser.add_argument("--num_particles", type=int, default=100, 
            help="Number of particles used for SVGD")
    parser.add_argument("--sgld", type=bool, default=True,
            help="Use sgld if true, else use svgd")

    args = parser.parse_args()

    # Simpler for now
    args.trajectory_length = args.prob_dim

    key = jax.random.PRNGKey(args.seed)
    train_key, test_key = jax.random.split(key, 2)

    args.problem_inst = importlib.import_module(f"problems.{args.problem}")

    phi_size = args.prob_dim * args.problem_inst.PHI_STATE_DIM

    in_size = args.latent_dim + phi_size
    out_size = args.trajectory_length * args.problem_inst.PHI_STATE_DIM

    model = ZDecoder(args.levels, args.regions, args.latent_dim, phi_size, out_size, key=jax.random.PRNGKey(0))

    print_error = lambda err, std: print(f"After training: Testing error: {err}, Testing STD: {std}")


    test_1_key, test_2_key = jax.random.split(key)
    print_error(*test(args, model, test_1_key))
    model = train(args, model, train_key)
    print_error(*test(args, model, test_2_key))

    eqx.tree_serialise_leaves(os.path.join(args.results_path, "model.eqx"),
            model)
