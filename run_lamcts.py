import numpy as np
import os
import json
from lamcts import MCTS
from toy_problem import get_toy_problem_functions
from argparse import ArgumentParser
import jax
from main import plot_solutions
from visuals import plot_single_problem

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
        trace_path = self.foldername + '/result' + str(len( self.results) )
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
        self.Cp        = 1
        self.leaf_size = 10
        self.ninits    = 10
        self.kernel_type = "rbf"
        self.gamma_type  = "auto"

    def __call__(self, x):
        print(x.shape, self.phi[0].shape)
        # Add dummy batch dimension
        x = np.expand_dims(x, axis=0)
        result = float(np.array(self.cost(x,self.phi)).squeeze())
        print(result)
        # result = (-20*np.exp(-0.2 * np.sqrt(np.inner(x,x) / x.size )) -np.exp(np.cos(2*np.pi*x).sum() /x.size) + 20 +np.e )
        self.tracker.track( result )
        return result
    
# f = myFunc()

def main(args):
    os.makedirs(args.results_path,exist_ok=True)
    samp_prob, get_phi, cost, mock_sol = get_toy_problem_functions(nwalls=2)
    key = jax.random.PRNGKey(args.seed)
    psi = samp_prob(key, 1)
    f = ToyProblemFunc(cost, psi, args=args)

    agent = MCTS(
                lb = f.lb,              # the lower bound of each problem dimensions
                ub = f.ub,              # the upper bound of each problem dimensions
                dims = f.dims,          # the problem dimensions
                ninits = f.ninits,      # the number of random samples used in initializations 
                func = f,               # function object to be optimized
                Cp = f.Cp,              # Cp for MCTS
                leaf_size = f.leaf_size, # tree leaf size
                kernel_type = f.kernel_type, #SVM configruation
                gamma_type = f.gamma_type    #SVM configruation
                )

    agent.search(iterations = 20)

    i=0
    phi = (psi[0][i], psi[1][i])
    q = agent.curt_best_sample
    print(q)
    q = np.expand_dims(q, axis=0)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plot_single_problem(fig, ax, phi, q[None, :], q.shape[0])
    fig.savefig(os.path.join(args.results_path, "plots.png"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_path", type=str, default="results/lamcts")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--rows", type=int, default=1, help="Number of columns in the plot grid")
    args = parser.parse_args()
    main(args)