import math
import os
import importlib
import pickle

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
        wkey, Zkey = jax.random.split(key, num=2)

        in_size = latent_dim + phi_size
        self.weight = jnp.ones((out_size, in_size))
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

        phi = jnp.broadcast_to(phi.reshape(batch_size, 1, dim), (batch_size, V_alphas.shape[0], dim))
        V_alphas = jnp.broadcast_to(V_alphas, (batch_size,)+V_alphas.shape)

        # batch, nregions*nlevels, z_size+phi_dim
        X = jnp.concatenate([V_alphas, phi], axis=-1) 
        qs = self.eval_mlp(X)
        return qs

def problem_dataloader(problems, batch_size):
    n_problems = problems[0].shape[0] # HACK
    n_batches_per_epoch = math.ceil(n_problems / batch_size)
    for batch_i in range(n_batches_per_epoch):
        if batch_i == n_batches_per_epoch-1:
            batch_probp = jax.tree_map(lambda x: x[batch_i*batch_size:], problems)
        else:
            batch_probp = jax.tree_map(lambda x: x[batch_i*batch_size: (batch_i+1)* batch_size], problems)
        yield batch_probp

def train(args, model, key, problems_train, problems_test, verbose=True):
    """
    Args:
        problems_train: tuple of all training batched problem params
        problems_test: 
    """
    _, get_phi, cost, mock_sol = args.problem_inst.make_problem(args.prob_dim)
    phi_size = args.prob_dim

    def compute_loss(model, phi, q_star):
        qs = model(phi)

        q_fit = (qs - q_star[:, None, :])**2
        q_fit = jnp.sum(q_fit, axis=-1)

        # inner minimization
        best_qs = jnp.min(q_fit, axis=1)
        # expectation
        loss = jnp.mean(best_qs, axis=0)
        return loss

    compute_train_loss = eqx.filter_value_and_grad(compute_loss)
    compute_test_loss = eqx.filter_jit(compute_loss)

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
        loss, grads = compute_train_loss(model, phi, q_star)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    # @ eqx.filter_jit
    # def make_step_sgld(qs, q_star, opt_state):
    #     loss, grads = compute_loss_q(qs, q_star)
    #     updates, opt_state = q_optim.update(grads, opt_state)
    #     qs = eqx.apply_updates(qs, updates)
    #     return qs, loss, opt_state

    optim = optax.adam(optax.exponential_decay(args.lr, 10, args.decay))
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    # TODO: early stopping?
    train_batch_size = args.problem_batch_size
    test_batch_size = args.test_batch_size
    for epoch in range(args.epochs):
        key, shuffle_key = jax.random.split(key)
        shuffle_problems_train = jax.tree_map(lambda x: jax.random.shuffle(shuffle_key, x), problems_train)
        for batch_probp in problem_dataloader(shuffle_problems_train, train_batch_size):
            key, key_solve = jax.random.split(key)
            phi = get_phi(batch_probp).reshape(train_batch_size, -1)
            q_star = mock_sol(key_solve, batch_probp).reshape(train_batch_size, -1)

            if args.sgld: # SGLD
                raise NotImplementedError()
                q_star = mock_sol(key_solve, probp)
                q_star = q_star.reshape(args.problem_batch_size, -1)
                qs = model(phi)
                q_optim = optax.noisy_sgd(args.lr, eta=0.005, gamma=0.85)
                q_opt_state = q_optim.init(qs)
                for iter in range(500):
                    qs, loss, q_opt_state = make_step_sgld(qs, q_star, q_opt_state)
                    if (iter + 1) % 100 == 0:
                        print(f"SGLD epoch={iter+1}, loss={loss : .4f}")
            elif args.svgd: # SVGD
                raise NotImplementedError()
                q_star = mock_sol(key_solve, probp)
                q_star = q_star.reshape(args.problem_batch_size, -1)
                svgd = svgd_utils.SVGD(model(phi), args.num_particles, args.seed)
                q_star = svgd.optimize(epochs=1000, gt=q_star, svgd_r=1)

            loss, model, opt_state = make_step(model, phi, q_star, opt_state)
            loss = loss.item()
        #losses.append(loss)
        if (epoch + 1) % 10 == 0:
            test_batch_count = 0
            avg_test_loss = 0.
            avg_test_cost = 0. # TODO
            for test_batch_probp in problem_dataloader(problems_test, test_batch_size):
                key, key_solve = jax.random.split(key)
                phi = get_phi(test_batch_probp).reshape(test_batch_size, -1)
                q_star = mock_sol(key_solve, test_batch_probp).reshape(test_batch_size, -1)
                avg_test_loss += compute_test_loss(model, phi, q_star)
                test_batch_count += 1
            avg_test_loss /= test_batch_count
            if verbose:
                print(f"epoch={epoch+1}, train_loss={loss : .4f} test_loss={avg_test_loss: .4f}")
            # TODO log to file

    return model

def eval(model, psi, cost, get_phi, state_dim):
    phi = get_phi(psi)
    batch_size = phi.shape[0]

    qh = model(phi)
    qh = qh.reshape(batch_size, qh.shape[1], -1, state_dim)
    c = cost(qh, psi)
    best_qs = c.argmin(axis=1)
    best_q = qh[jnp.arange(batch_size), best_qs]
    return best_q, c[jnp.arange(batch_size),best_qs]

def test(args, model, key, problems_test):
    samp_prob, get_phi, cost, mock_sol = args.problem_inst.make_problem(args.prob_dim)
    test_batch_size = args.test_batch_size
    test_batch_count = 0
    avg_test_cost = 0. # TODO
    for test_batch_probp in problem_dataloader(problems_test, test_batch_size):
        # key, key_solve = jax.random.split(key)
        # phi = get_phi(test_batch_probp).reshape(test_batch_size, -1)
        # q_star = mock_sol(key_solve, test_batch_probp).reshape(test_batch_size, -1)

        key_sample, key_solve = jax.random.split(key)
        psi = samp_prob(key_sample, test_batch_size)
        # gt = mock_sol(key_solve, psi)
        # phi = get_phi(psi)
        # qs = model(phi)
        best_q, best_q_cost = eval(model, psi, cost, get_phi, args.problem_inst.PHI_STATE_DIM)
        print('best_q_cost', best_q_cost.shape)

        avg_test_cost += jnp.mean(best_q_cost, axis=0).item()
        test_batch_count += 1
    avg_test_cost /= test_batch_count
    # err = jnp.linalg.norm(gt - best_q, axis=-1)
    # cost(best_q)
    # c = cost(qh, psi)

    # mean_error = jnp.mean(err)
    # std_dev = jnp.std(err)

    # if args.plot:
    #     raise NotImplementedError()
    #     plot_solutions(args, psi, gt, qs, args.results_path)

    # return mean_error, std_dev
    return avg_test_cost

def get_data(args, train_data_size, test_data_size, key, cache_path=None):
    if cache_path is not None and os.path.exists(cache_path):
        raise NotImplementedError()
    else:
        samp_prob, _, _, _ = args.problem_inst.make_problem(args.prob_dim)
        data_train_key, data_test_key = jax.random.split(key, 2)
        train_probp = samp_prob(data_train_key, batch_size=train_data_size)
        test_probp = samp_prob(data_test_key, batch_size=train_data_size)

        if cache_path is not None:
            raise NotImplementedError()
            with open(cache_path, 'wb') as f:
                pickle.dump(probp, f)
    return train_probp, test_probp

def main(args):
    # Simpler for now
    args.trajectory_length = args.prob_dim
    args.problem_inst = importlib.import_module(f"problems.{args.problem}")

    key = jax.random.PRNGKey(args.seed)
    key, data_key = jax.random.split(key)

    train_data_size = 1000
    test_data_size = 200
    probp_train, probp_test = get_data(args, train_data_size, test_data_size, data_key)

    phi_size = args.prob_dim * args.problem_inst.PHI_STATE_DIM
    out_size = args.trajectory_length * args.problem_inst.PHI_STATE_DIM

    n_fracs = 10
    fracs = [(1.+i)/n_fracs for i in range(n_fracs)]
    train_sizes = [min(round(frac*train_data_size),train_data_size) for frac in fracs]
    print(train_sizes)
    avg_test_costs = []
    for train_size in train_sizes:
        # Train new model with different fraction of training data
        model = ZDecoder(args.levels, args.regions, args.latent_dim, phi_size, out_size, key=jax.random.PRNGKey(0))

        probp_train_frac = jax.tree_map(lambda x: x[:train_size], probp_train)
        model = train(args, model, key, probp_train_frac, probp_test)
        avg_test_cost = test(args,model, key, probp_test)
        avg_test_costs.append(avg_test_cost)

    print(train_sizes, avg_test_costs)
    fig, ax = plt.subplots()
    ax.plot(train_sizes, avg_test_costs)
    fig.savefig(os.path.join("gen_perf.png"))
        

    # print_error = lambda err, std: print(f"After training: Testing error: {err}, Testing STD: {std}")
    # test_1_key, test_2_key = jax.random.split(key)
    # print_error(*test(args, model, test_1_key))
    # model = train(args, model, train_key)
    # print_error(*test(args, model, test_2_key))

    eqx.tree_serialise_leaves(os.path.join(args.results_path, "model.eqx"),
            model)


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--problem", type=str, default="toy_problem")

    parser.add_argument("--problem_batch_size", type=int, default=50,
            help="For each iteration how many problems to sample")
    parser.add_argument("--epochs", type=int, default=100,
            help="Total iteration count")

    parser.add_argument("--levels", type=int, default=8,
            help="Number of levels to the problem,"
            + "can't be greater than latent dimension")
    parser.add_argument("--regions", type=int, default=2,
            help="Number of voronoi regions per level")

    parser.add_argument("--test_batch_size", type=int, default=1000,
            help="For testing how many problems to sample")
    parser.add_argument("--latent_dim", type=int, default=8,
            help="Dimensions in the latent space")
    parser.add_argument("--prob_dim", type=int, default=8,
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
    parser.add_argument("--sgld", type=bool, default=False,
            help="Use sgld if true")
    parser.add_argument("--svgd", type=bool, default=False,
            help="Use svgd if true")

    args = parser.parse_args()
    main(args)