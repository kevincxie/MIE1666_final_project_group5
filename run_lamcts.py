import numpy as np
import os
import json
from lamcts import MCTS
from problems.toy_problem import make_problem, plot_single_problem, get_traj_length
from argparse import ArgumentParser
import jax
#from main import plot_solutions

class tracker:
    def __init__(self, foldername):
        self.counter   = 0
        self.results   = []
        self.curt_best = float("inf")
        self.foldername = foldername
        try:
            os.mkdir(foldername)
        except OSError:
            print ("Creation of the directory %s failed" % foldername)
        else:
            print ("Successfully created the directory %s " % foldername)
        
    def dump_trace(self):
        trace_path = self.foldername + str(len( self.results) )
        final_results_str = json.dumps(self.results)
        with open(trace_path, "a") as f:
            f.write(final_results_str + '\n')
            
    def track(self, result):
        if result < self.curt_best:
            self.curt_best = result
        self.results.append(self.curt_best)
        if len(self.results) % 100 == 0:
            self.dump_trace()

class ToyProblemFunc:
    def __init__(self, cost, phi, dims=2, args=None):
        self.cost = cost
        self.phi = phi
        self.dims    = dims                   #problem dimensions
        # self.lb      =  np.ones(dims)         #lower bound for each dimensions 
        # self.ub      =  np.ones(dims)         #upper bound for each dimensions 
        self.lb        = -2 * np.ones(dims)
        self.ub        =  2 * np.ones(dims)
        if args:
            self.tracker = tracker(args.results_path)      #defined in functions.py

        #tunable hyper-parametuuers in LA-MCTS
        self.Cp        = 10
        self.leaf_size = 10
        self.ninits    = 40
        self.kernel_type = "rbf"
        self.solver_type             = 'turbo' #solver can be 'bo' or 'turbo'

        self.gamma_type  = "auto"

    def __call__(self, x):
        # print(x.shape, self.phi[0].shape)
        # Add dummy batch dimension
        # x = x.reshape((1,1,8,1))
        result = float(np.array(self.cost(x,self.phi)).squeeze())
        self.tracker.track( result )
        return result

def main(args):
    os.makedirs(args.results_path,exist_ok=True)
    connecting_steps = 10
    n_walls = 4
    ndims = get_traj_length(n_walls, connecting_steps)
    samp_prob, get_phi, cost, mock_sol = make_problem(nwalls=n_walls, connecting_steps=connecting_steps)
    key = jax.random.PRNGKey(args.seed)
    psi = samp_prob(key, 1)
    psi = jax.tree_map(lambda x: x.squeeze(0), samp_prob(key, 1))
    opt_cost = cost(np.expand_dims(mock_sol(None, psi),axis=0), psi)
    f = ToyProblemFunc(cost, psi, dims=ndims, args=args)

    turbo_steps = 100

    agent = MCTS(
                lb = f.lb,              # the lower bound of each problem dimensions
                ub = f.ub,              # the upper bound of each problem dimensions
                dims = f.dims,          # the problem dimensions
                ninits = f.ninits,      # the number of random samples used in initializations 
                func = f,               # function object to be optimized
                Cp = f.Cp,              # Cp for MCTS
                leaf_size = f.leaf_size, # tree leaf size
                kernel_type = f.kernel_type, #SVM configruation
                gamma_type = f.gamma_type,    #SVM configruation
                solver_type = f.solver_type,
                turbo_steps= turbo_steps
                )

    agent.search(iterations = 1000)

    i=0
    # phi = (psi[0][i], psi[1][i])
    phi = (psi[0], psi[1])
    q = agent.curt_best_sample
    print(q)
    q = np.expand_dims(q, axis=0)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # plot_single_problem(fig, ax, phi, q[None, :], q.shape[0])
    plot_single_problem(fig, ax, phi, q[None, :], connecting_steps=connecting_steps, modes=q.shape[0],)
    fig.savefig(os.path.join(args.results_path, "plots.png"))

    fig, ax = plt.subplots()
    ax.plot(f.tracker.results, label='LAMCTS')
    ax.axhline(opt_cost, c='red', label='Optimal')
    fig.savefig(os.path.join(args.results_path, "perf_plots.png"))



if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--results_path", type=str, default="results/lamcts/")
    parser.add_argument("--results_path", type=str, default="results/lamcts/turbo")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--rows", type=int, default=1, help="Number of columns in the plot grid")
    args = parser.parse_args()
    main(args)
