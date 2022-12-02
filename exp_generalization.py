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

from main import ZDecoder
from main import plot_solutions


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
    _, get_phi, cost, mock_sol = args.problem_inst.make_problem(args.n_walls, args.connecting_steps)
    mock_sol = jax.vmap(mock_sol, (None,0))
    phi_size = args.n_walls

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
    # qh = qh.reshape(batch_size, qh.shape[1], -1, state_dim)
    c = jax.vmap(jax.vmap(cost, (0,None),out_axes=0),(0,0), out_axes=0)(qh, psi)
    best_qs = c.argmin(axis=1)
    best_q = qh[jnp.arange(batch_size), best_qs]
    return best_q, c[jnp.arange(batch_size),best_qs]

def test(args, model, key, problems_test):
    samp_prob, get_phi, cost, mock_sol = args.problem_inst.make_problem(args.n_walls, args.connecting_steps)
    test_batch_size = args.test_batch_size
    test_batch_count = 0
    avg_test_cost = 0. # TODO
    for test_batch_probp in problem_dataloader(problems_test, test_batch_size):
        # key, key_solve = jax.random.split(key)
        # phi = get_phi(test_batch_probp).reshape(test_batch_size, -1)
        # q_star = mock_sol(key_solve, test_batch_probp).reshape(test_batch_size, -1)

        # key_sample, key_solve = jax.random.split(key)
        # psi = samp_prob(key_sample, test_batch_size)
        # gt = mock_sol(key_solve, psi)
        # phi = get_phi(psi)
        # qs = model(phi)
        best_q, best_q_cost = eval(model, test_batch_probp, cost, get_phi, args.problem_inst.PHI_STATE_DIM)

        avg_test_cost += jnp.mean(best_q_cost, axis=0).item()
        test_batch_count += 1
    avg_test_cost /= test_batch_count
    # err = jnp.linalg.norm(gt - best_q, axis=-1)
    # cost(best_q)
    # c = cost(qh, psi)

    # mean_error = jnp.mean(err)
    # std_dev = jnp.std(err)

    if args.plot:
        psi = samp_prob(key, 1)
        gt = jax.vmap(mock_sol, (None,0))(key, test_batch_probp)[0]
        # phi = get_phi(psi)
        # qs = model(phi)
        q_star = best_q[0]
        print(q_star.shape)
        os.makedirs(args.results_path, exist_ok=True)
        # plot_solutions(args, psi, gt, qs[0], args.results_path, args.connecting_steps)

        fig, ax = plt.subplots()
        plot_single_problem = args.problem_inst.plot_single_problem

        probp = jax.tree_map(lambda x: x[0], test_batch_probp)
        plot_single_problem(fig, ax, probp, q_star[None, :], connecting_steps=args.connecting_steps)
        plot_single_problem(fig, ax, probp, gt[None, :], connecting_steps=args.connecting_steps)
        fig.savefig(os.path.join(args.results_path, 'viz_best.png'))

    # if args.plot:
    #     raise NotImplementedError()
    #     plot_solutions(args, psi, gt, qs, args.results_path)

    # return mean_error, std_dev
    return avg_test_cost

def get_data(args, train_data_size, test_data_size, key, cache_path=None):
    if cache_path is not None and os.path.exists(cache_path):
        raise NotImplementedError()
    else:
        samp_prob, _, _, _ = args.problem_inst.make_problem(args.n_walls, args.connecting_steps)
        data_train_key, data_test_key = jax.random.split(key, 2)
        train_probp = samp_prob(data_train_key, batch_size=train_data_size)
        test_probp = samp_prob(data_test_key, batch_size=test_data_size)

        if cache_path is not None:
            raise NotImplementedError()
            with open(cache_path, 'wb') as f:
                pickle.dump(probp, f)
    return train_probp, test_probp

def get_oracle_perf(args, key, probp_test):
    _, _, cost, mock_sol = args.problem_inst.make_problem(args.n_walls, args.connecting_steps)
    q_oracle = jax.vmap(mock_sol,(None,0))(key, probp_test)
    print(q_oracle.shape)
    return jnp.mean(cost(q_oracle, probp_test), axis=0)

def main(args):
    # Simpler for now
    args.n_walls = args.n_walls
    args.problem_inst = importlib.import_module(f"problems.{args.problem}")

    # HACK: hard_coding this
    args.connecting_steps = 1
    args.trajectory_length = args.n_walls + args.connecting_steps * (args.n_walls-1)

    key = jax.random.PRNGKey(args.seed)
    key, data_key = jax.random.split(key)


    train_data_size = 500
    test_data_size = 2000
    probp_train, probp_test = get_data(args, train_data_size, test_data_size, data_key)
    assert probp_train[0].shape[0] == train_data_size
    assert probp_test[0].shape[0] == test_data_size

    phi_size = args.n_walls * args.problem_inst.PHI_STATE_DIM
    out_size = (args.trajectory_length * args.problem_inst.PHI_STATE_DIM, )

    # Compute oracle performance on mock solutions
    oracle_cost = get_oracle_perf(args, key, probp_test).item()
    print(f"Oracle cost {oracle_cost : .4f}")

    def run_model_data_ablation(model_constructor, train_sizes):
        avg_test_costs = []
        for train_size in train_sizes:
            # Train new model with different fraction of training data
            # model = ZDecoder(args.levels, args.regions, args.latent_dim, phi_size, out_size, key=jax.random.PRNGKey(0))
            model = model_constructor()

            probp_train_frac = jax.tree_map(lambda x: x[:train_size], probp_train)
            model = train(args, model, key, probp_train_frac, probp_test)
            avg_test_cost = test(args,model, key, probp_test)
            avg_test_costs.append(avg_test_cost)
        return {'train_sizes':train_sizes, 'avg_test_costs': avg_test_costs}

    n_fracs = 10
    fracs = [(1.+i)/n_fracs for i in range(n_fracs)]
    train_sizes = [min(round(frac*train_data_size),train_data_size) for frac in fracs]

    perf_per_model = {}
    # perf_per_model['oracle'] = oracle_cost
    print('Training no_decoder')
    if args.connecting_steps == 0:
        perf_per_model['no_decoder'] = run_model_data_ablation(
            lambda: ZDecoder(levels=args.n_walls, regions=args.regions, 
            latent_dim=args.n_walls, phi_size=phi_size, out_shape=out_size, key=key,
            identity_decoder=True), train_sizes)
    print('Training full')
    args.plot = True
    args.results_path = 'gen_perf_traj_full'
    perf_per_model['full'] = run_model_data_ablation(
        lambda: ZDecoder(levels=args.n_walls, regions=args.regions, 
        latent_dim=args.n_walls, phi_size=phi_size, out_shape=out_size, key=key,
        identity_decoder=False), train_sizes)
    args.plot = False
    print('Training no_Z')
    perf_per_model['no_Z'] = run_model_data_ablation(
        lambda: ZDecoder(levels=args.n_walls, regions=1, 
        latent_dim=args.n_walls, phi_size=phi_size, out_shape=out_size, key=key,
        identity_decoder=False), train_sizes)
        
    print(perf_per_model)


    fig, ax = plt.subplots()
    ax.set_title('Generalization Performance Ablations')
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('Cost')
    for model_name, perfd in perf_per_model.items():
        ax.plot(perfd['train_sizes'], perfd['avg_test_costs'], label=model_name)
    ax.axhline(oracle_cost, label='oracle', c='red')
    ax.legend()
    fig.savefig(os.path.join("gen_perf.png"))
        

    # print_error = lambda err, std: print(f"After training: Testing error: {err}, Testing STD: {std}")
    # test_1_key, test_2_key = jax.random.split(key)
    # print_error(*test(args, model, test_1_key))
    # model = train(args, model, train_key)
    # print_error(*test(args, model, test_2_key))
    # eqx.tree_serialise_leaves(os.path.join(args.results_path, "model.eqx"),
    #         model)


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--problem", type=str, default="toy_problem")

    parser.add_argument("--problem_batch_size", type=int, default=50,
            help="For each iteration how many problems to sample")
    parser.add_argument("--epochs", type=int, default=100,
            help="Total iteration count")

    # parser.add_argument("--levels", type=int, default=8,
    #         help="Number of levels to the problem,"
    #         + "can't be greater than latent dimension")
    parser.add_argument("--regions", type=int, default=2,
            help="Number of voronoi regions per level")

    parser.add_argument("--test_batch_size", type=int, default=1000,
            help="For testing how many problems to sample")
    # parser.add_argument("--latent_dim", type=int, default=8,
    #         help="Dimensions in the latent space")
    parser.add_argument("--n_walls", type=int, default=8,
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
