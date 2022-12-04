import numpy as np
import os
import json
from skopt import gp_minimize
from turbo_1.turbo_1 import Turbo1
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

        #tunable hyper-parameters in LA-MCTS
        self.Cp        = 10
        self.leaf_size = 10
        self.ninits    = 40
        self.kernel_type = "rbf"
        self.gamma_type  = "auto"

    def __call__(self, x):
        # print(x)
        x = np.array(x)
        # print(x.shape, self.phi[0].shape)
        # Add dummy batch dimension
        #x = x.reshape((8,))
        result = float(np.array(self.cost(x,self.phi)).squeeze())
        self.tracker.track( result )
        print(result)
        return result

def main(args):
    os.makedirs(args.results_path,exist_ok=True)
    n_walls = 4
    connecting_steps = 10
    ndims = get_traj_length(n_walls, connecting_steps)
    samp_prob, get_phi, cost, mock_sol = make_problem(nwalls=n_walls, connecting_steps=connecting_steps)
    key = jax.random.PRNGKey(args.seed)
    psi = jax.tree_map(lambda x: x.squeeze(0), samp_prob(key, 1))
    opt_cost = cost(np.expand_dims(mock_sol(None, psi),axis=0), psi)
    f = ToyProblemFunc(cost, psi, dims=ndims, args=args)

    # agent = MCTS(
    #             lb = f.lb,              # the lower bound of each problem dimensions
    #             ub = f.ub,              # the upper bound of each problem dimensions
    #             dims = f.dims,          # the problem dimensions
    #             ninits = f.ninits,      # the number of random samples used in initializations 
    #             func = f,               # function object to be optimized
    #             Cp = f.Cp,              # Cp for MCTS
    #             leaf_size = f.leaf_size, # tree leaf size
    #             kernel_type = f.kernel_type, #SVM configruation
    #             gamma_type = f.gamma_type    #SVM configruation
    #             )

    # res = gp_minimize(f,                  # the function to minimize
    #               [(-2.0, 2.0)]*ndims,      # the bounds on each dimension of x
    #               acq_func="EI",      # the acquisition function
    #               n_calls=100,         # the number of evaluations of f
    #               n_random_starts=5,  # the number of random initialization points
    #               noise=0.1**2,       # the noise level (optional)
    #               random_state=1234)   # the random seed
    # x, fun = res.x, res.fun 
    # X_init = 
    num_samples = 2000
    turbo1 = Turbo1(
        f  = f,              # Handle to objective function
        lb = f.lb,           # Numpy array specifying lower bounds
        ub = f.ub,           # Numpy array specifying upper bounds
        n_init = 30,            # Number of initial bounds from an Latin hypercube design
        max_evals  = num_samples, # Maximum number of evaluations
        batch_size = 1,         # How large batch size TuRBO uses
        verbose=True,           # Print information from each batch
        use_ard=True,           # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=2000, # When we switch from Cholesky to Lanczos
        n_training_steps=50,    # Number of steps of ADAM to learn the hypers
        min_cuda=1024,          #  Run on the CPU for small datasets
        device="cpu",           # "cpu" or "cuda"
        dtype="float32",        # float64 or float32
        X_init = np.random.uniform(size = (1, n_walls *(connecting_steps+1))),
    )
    
    x, fun = turbo1.optimize( )


    # agent.search(iterations = 100)
    # x, fun = proposed_X, 
    print(f"{x} {fun}")
    # "x^*=%.4f, f(x^*)=%.4f" % (res.x, res.fun)


    i=0
    phi = (psi[0], psi[1])
    # q = agent.curt_best_sample
    q = np.array(x)
    print(q)
    q = np.expand_dims(q, axis=0)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plot_single_problem(fig, ax, phi, q[None, :], connecting_steps=connecting_steps, modes=q.shape[0],)
    fig.savefig(os.path.join(args.results_path, "plots.png"))

    fig, ax = plt.subplots()
    ax.plot(f.tracker.results, label='BO')
    ax.axhline(opt_cost, c='red', label='Optimal')
    fig.savefig(os.path.join(args.results_path, "perf_plots.png"))



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_path", type=str, default="results/turbo/")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--rows", type=int, default=1, help="Number of columns in the plot grid")
    args = parser.parse_args()
    main(args)
