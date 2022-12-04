import os
import importlib

import numpy as np

import jax
from jax import random
import jax.numpy as jnp
import optax
import equinox as eqx
import equinox.nn as nn
import pdb

from jaxtyping import Array

import matplotlib.pyplot as plt
import seaborn as sns

from argparse import ArgumentParser

from typing import Callable, Tuple
from functools import partial

import problems

from ops.common import problem_dataloader

import ops.common
import svgd_utils

class ZDecoder(eqx.Module):
#    weight: jnp.ndarray
    #bias: jnp.ndarray
    region_params: jnp.ndarray

    particle_confidence: jnp.ndarray
    train: bool = False

    model: nn.MLP

    z_size: int
    levels: int
    regions: int
    out_shape: Tuple[int]
    identity_decoder: bool

    def __init__(self, levels, regions, latent_dim, phi_size, out_size, key, identity_decoder=False):
        assert(levels <= latent_dim and latent_dim % levels == 0)
        self.z_size = latent_dim // levels
        self.levels = levels
        self.regions = regions
        self.out_shape = (out_size,)
        self.identity_decoder = identity_decoder
        in_size = 2*latent_dim // levels + phi_size #// args.levels

        wkey, bkey, Zkey = jax.random.split(key, num=3)
#       self.weight2 = jax.random.normal(bkey, (out_size, phi_size))
#       self.weight = jax.random.uniform(wkey, (out_size, in_size))
        self.region_params = jax.random.normal(Zkey, (levels, regions, self.z_size))
        self.particle_confidence = jnp.ones(regions**levels)

        if self.identity_decoder:
            # Ignore the
            self.model = lambda x, phi: x
        else:
            self.model = nn.MLP(in_size + levels, out_size, 64, 2, key=wkey)

    @partial(jax.vmap, in_axes=(None, None, 0, None), out_axes=0) # Batch
    @partial(jax.vmap, in_axes=(None, 0, None, None), out_axes=0) # r^l
    @partial(jax.vmap, in_axes=(None, 0, None, 0), out_axes=0) # Levels
#    @partial(jax.vmap, in_axes=(None, 0, None, None), out_axes=0) # Regions
    def eval_mlp(self, x, phi, idx_vec):
        # print(x.shape, phi.shape, idx_vec.shape)
        return self.model(jnp.concatenate([x, phi, idx_vec], axis=-1))

    @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    def build_regions(self, X):
        i = jnp.mgrid[(slice(0, self.regions, 1),)*self.levels]
        V_alphas = X[jnp.arange(self.levels), i.T].reshape(-1, self.levels*self.z_size, X.shape[-1])
        return V_alphas

    def resample(self, key):
        threshold = int(self.particle_confidence.shape[0] / self.regions**4)
        indices = jnp.argwhere(self.particle_confidence < 0.01).T
        #print(self.particle_confidence)

        region_array = jnp.mgrid[:self.levels, :self.regions].transpose(1, 2, 0)
        indexed = self.build_regions(region_array[None])[0]
        culled_region_indices = indexed[indices]
        # print(culled_region_indices.shape)
        if culled_region_indices.shape[1] > 0:
         #   print(culled_region_indices)
            new_regions = jax.random.normal(key, shape=(culled_region_indices.shape[0],1))
            region_params = self.region_params.at[culled_region_indices[:, 0], culled_region_indices[:, 1]].set(new_regions)
            particle_confidence = self.particle_confidence.at[indices].set(1.)

            self = eqx.tree_at(lambda x: (x.region_params, x.particle_confidence), self, (region_params, particle_confidence))
        return self
      #  return new_region_params, new_particle_confidence

    def __call__(self, phi):
        phi = phi.reshape(phi.shape[0], -1) # Treat all the enviorenment parameters as one long vector..
        batch_size, dim = phi.shape

        # print(X.shape)
        idx_vecs = jnp.eye(self.levels) # Batch, regions**levels, 1

        zs = self.build_regions(self.region_params[None])[0]
        # print(zs.shape)
        future = jnp.roll(zs, -1, axis=1)
        x = jnp.concatenate([zs, future], axis=-1)

        qs = self.eval_mlp(x, phi, idx_vecs) # [Batch, levels, regions, q_size]
#        trajectories = self.build_regions(qs)
#        trajectories = trajectories.reshape(batch_size, trajectories.shape[1], -1)
        # print(qs.shape)

        return qs.reshape(batch_size, qs.shape[1], -1)

def get_optimizer(args):
    opt = getattr(ops.common, args.optimizer)
    return opt(args.optim_step_size, **vars(args))

def eval(model, psi, cost, get_phi, state_dim):
    phi = get_phi(psi)
    batch_size = phi.shape[0]

    qh = model(phi)
    qh = qh.reshape(batch_size, qh.shape[1], -1)
    c = jax.vmap(cost, in_axes=0, out_axes=0)(qh, psi) # [ntrajs, nwalls]

    best_qs = c.argmin(axis=1)
    best_q = qh[jnp.arange(batch_size), best_qs]
    return best_q

def train(args, optimizer, model, key, problems_train=None):
    samp_prob, get_phi, cost, mock_sol = args.problem_inst.make_problem(args.n_walls, args.connecting_steps)
    phi_size = args.n_walls
    train_batch_size = args.problem_batch_size
    in_size, out_size = args.latent_dim + phi_size, args.n_walls

    @eqx.filter_value_and_grad
    def compute_loss(model, phi, q_star, weights):
        qs = model(phi)
        q_fit = (qs[:, None, :] - q_star[:, :, None, :])**2
        q_fit = jnp.sum(q_fit, axis=-1)
        best_qs = jnp.argmin(q_fit, axis=2)


        # inner minimization
        best_qs = q_fit.min(axis=2)
        best_qs *= weights

        # expectation
        loss = jnp.mean(best_qs, axis=(1, 0))
        return loss

    def calculate_confidence(model, phi, q_star, weights):
        qs = model(phi)
        q_fit = (qs[:, None, :] - q_star[:, :, None, :])**2
        q_fit = jnp.sum(q_fit, axis=-1)
        best_qs = jnp.argmin(q_fit, axis=2)
        weights = jnp.clip(weights, a_min=0.1)
        confidence = jax.nn.one_hot(best_qs, q_fit.shape[2]) * weights[..., None]
        confidence = confidence.sum(axis=(0, 1))
        return confidence

    # Important for efficiency whenever you use JAX: wrap everything into a single JIT
    # region.
    @eqx.filter_jit
    def make_step(model, phi, q_star, opt_state, weights):
        loss, grads = compute_loss(model, phi, q_star, weights)
        updates, opt_state = optim.update(grads, opt_state)

        confidence = calculate_confidence(model, phi, q_star, weights)
        model = eqx.apply_updates(model, updates)
        model = eqx.tree_at(lambda x: x.particle_confidence, model, 0.7 * confidence + 0.3 * model.particle_confidence)

        return loss, model, opt_state

    optim = optax.adam(optax.exponential_decay(args.lr, 10, args.decay))
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    for epoch in range(args.epochs):
        if problems_train is None:
            # If we don't have dataset, each epoch is just a random sampled batch
            key, key_sample = jax.random.split(key,2)
            probp = samp_prob(key_sample, batch_size=train_batch_size)
            epoch_problems = [probp]
        else:
            key, shuffle_key = jax.random.split(key)
            shuffle_problems_train = jax.tree_map(lambda x: jax.random.shuffle(shuffle_key, x), problems_train)
            epoch_problems = problem_dataloader(shuffle_problems_train, train_batch_size)

        for batch_probp in epoch_problems:
            key, key_solve = jax.random.split(key, 2)
            phi = get_phi(batch_probp).reshape(train_batch_size, -1)
            qhat = model(phi)

            qn = jax.random.uniform(key_solve, shape=(phi.shape[0], args.num_particles, phi.shape[1]*(args.connecting_steps+1)))
            qhat = jnp.concatenate([qhat, qn], axis=1)


            def optimize(x, psi):
                q_star, c, _ = optimizer(key_solve, range(args.iterations), lambda xh: cost(xh, psi), x)
                return q_star, c

            q_star, c = jax.vmap(jax.jit(optimize), in_axes=0, out_axes=0)(qhat, batch_probp)
            weights = jnp.exp(jax.nn.log_softmax(-c, axis=1))

            # mean = c.mean(axis=(0, 1))
            # std = c.std(axis=(0, 1))
            # print(mean, std)

            loss, model, opt_state = make_step(model, phi, q_star, opt_state, weights)

            loss = loss.item()

        if (epoch + 1) % 5 == 0:
            key, key_resample = random.split(key)
            model = model.resample(key_resample)
            key, key_test = random.split(key)
            test_err, test_std = test(args, model, key)
            print(f"epoch={epoch+1}, loss={loss : .4f} test_err={test_err : .5f} test_std={test_std : .5f}")

    return model


def plot_solutions(args, psi, gt, qs, path, connecting_steps):
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
        args.problem_inst.plot_single_problem(fig, ax, phi, q[None, :], args.connecting_steps, q.shape[0])
  #      args.problem_inst.plot_single_problem(fig, ax, phi, gt_.reshape(1, 1, -1), 1)
    if not args.plot_once:
        args.iter += 1
    os.makedirs(os.path.join(path, f"results/"), exist_ok=True)
    fig.savefig(os.path.join(path, f"results/plots{args.iter}.png"))

def test(args, model, key):
    test_batch_size = 1 # args.test_batch_size
    samp_prob, get_phi, cost, mock_sol = args.problem_inst.make_problem(args.n_walls, args.connecting_steps)
    key_sample, key_solve = jax.random.split(key)
    #key_sample = key_solve = key
    psi = samp_prob(key_sample, test_batch_size)
    gt = jax.vmap(mock_sol, in_axes=(None, 0), out_axes=0)(key_solve, psi)
    phi = get_phi(psi)
    qs = model(phi)

    best_q = eval(model, psi, cost, get_phi, args.problem_inst.PHI_STATE_DIM)
    err = jnp.linalg.norm(gt - best_q, axis=-1)

    mean_error = jnp.mean(err)
    std_dev = jnp.std(err)

    print(psi[0].shape, gt.shape, qs.shape)
    if args.plot:
        plot_solutions(args, psi, gt, qs, args.results_path, args.connecting_steps)

    return mean_error, std_dev


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--problem", type=str, default="toy_problem")

    parser.add_argument("--problem_batch_size", type=int, default=10,
            help="For each iteration how many problems to sample")
    parser.add_argument("--epochs", type=int, default=100,
            help="Total iteration count")

    parser.add_argument("--levels", type=int, default=-1,
            help="Number of levels to the problem,"
            + "can't be greater than latent dimension")
    parser.add_argument("--regions", type=int, default=2,
            help="Number of voronoi regions per level")

    parser.add_argument("--test_batch_size", type=int, default=1,
            help="For testing how many problems to sample")
    parser.add_argument("--latent_dim", type=int, default=-1,
            help="Dimensions in the latent space")

    parser.add_argument("--connecting_steps", type=int, default=2,
            help="Connections between walls")
    parser.add_argument("--n_walls", type=int, default=2,
            help="Number of dimensions in the problem," +
            "corresponds to the number of walls")

    parser.add_argument("--plot", action='store_true', default=False)
    parser.add_argument("--rows", type=int, default=1, help="Number of columns in the plot grid")
    parser.add_argument("--plot_height", type=int, default=6)
    parser.add_argument("--plot_width", type=int, default=8)

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--decay", type=float, default=1.)

    parser.add_argument("--results_path", type=str, default=".")

    subparsers = parser.add_argument("--optimizer", type=str, default='sgld')

    parser.add_argument("--num_particles", type=int, default=50,
            help="Number of particles used for SVGD")

    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.75)
    parser.add_argument("--eta", type=float, default=0.01)

    parser.add_argument("--optim-step-size", type=float, default=0.01)

    parser.add_argument("--plot-once", action='store_true', default=True)

    args = parser.parse_args()

    if args.latent_dim < 0:
        args.latent_dim = args.n_walls

    if args.levels < 0:
        args.levels = args.n_walls

    args.trajectory_length = args.n_walls + args.connecting_steps * args.n_walls

    key = jax.random.PRNGKey(args.seed)
    model_key, train_key, test_key = jax.random.split(key, 3)

    args.problem_inst = importlib.import_module(f"problems.{args.problem}")
    args.iter = 0

    phi_size = args.n_walls * args.problem_inst.PHI_STATE_DIM

    in_size = 2*args.latent_dim // args.levels + phi_size #// args.levels
    out_size = args.trajectory_length // args.n_walls

    model = ZDecoder(args.levels, args.regions, args.latent_dim, phi_size, out_size, key=model_key)

    print_error = lambda err, std: print(f"After training: Testing error: {err}, Testing STD: {std}")

    optimizer = get_optimizer(args)

    test_1_key, test_2_key = jax.random.split(test_key)
    print_error(*test(args, model, test_1_key))
    model = train(args, optimizer,  model, train_key)
    print_error(*test(args, model, test_2_key))

    eqx.tree_serialise_leaves(os.path.join(args.results_path, "model.eqx"),
            model)
