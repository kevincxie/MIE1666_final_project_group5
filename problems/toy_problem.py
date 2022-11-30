from typing import Tuple
import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial

from .visuals import plot_background, plot_solution
import problems

PHI_STATE_DIM = 1 # Size of the problem desc per dimension

def plot_single_problem(fig, ax, phi, soln, modes):
    plot_background(fig, ax, phi, phi[0].shape[0], phi[1].shape[1], wall_width_pct=0.25, wall_height_pct=0.7)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    for i in range(modes):
        plot_solution(ax, soln[:, i, :])

    ax.tick_params(which='both', bottom=False, top=False, labelbottom=False, labelleft=False)

def make_problem(nwalls=2):
    # Represents ndim walls, each with 2 holes, q is ndim long and each dim
    # corresponds to a wall
    # This is [nwalls, nholes_per_wall]
    q_holes = jnp.array([[-1.,1.]]*nwalls)
    nholes_per_wall = 2

    def get_problem_phi(params):
        """
        This is the observable context of the problem
        """
        # Just expose the shift
        return params[0]

    def sample_problem_params(key, batch_size) -> Tuple:
        """
        Args:
            key: jax RNG key
        Returns:
            (phi): Tuple of params
        """
        # Small shift in location per wall
        phi_shift = jax.random.uniform(key, shape=(batch_size,nwalls,), minval=-0.5, maxval=0.5)

        # Weigh each hole differently
        phi_weight = jax.random.uniform(key, shape=(batch_size,nwalls,nholes_per_wall), minval=0.1, maxval=1.0)
        return (phi_shift, phi_weight)

    @partial(jnp.vectorize, signature='(s,1),(s,r)->(s,r)')
    def gaussian_cost_1d(x, center):
        return -jnp.exp(-((x-center)*2.)**2)

    def cost(q, prob_params):
        """
        Cost function computation
        Args:
            q: (nwalls,)
            prob_params: [(nwalls, ), (nwalls, nholes)]
        Returns:
            cost 
        """
        phi_shift, phi_weight = prob_params
        # for a single q, calculate its cost to all holes
        # ptim.update()

        # Add dummy "holes" broadcast dimension to q
        q = jnp.expand_dims(q,-1)

        # get the shape of phi to be [batch, *, phi_dim] where * is arbitary dims in q
        shifted_holes = q_holes+phi_shift[..., None]
        
        cost = gaussian_cost_1d(q, shifted_holes)
        # multiply each hole by weight
        cost = cost * phi_weight

        # Sum over all holes on each wall
        return jnp.sum(cost, (-2,-1)) 

    # Batch over problem params
    def mock_solution(key, prob_params):
        """
        Pretends to do VOO
        Return:
            q_star: Approximate solution to phi
        """
        # Get the hole for each wall with the highest weight
        phi_shift, phi_weight = prob_params
        assert(phi_shift.ndim == 1)
        assert(phi_weight.ndim == 2)

        best_hole = jnp.argmax(phi_weight, axis=-1)

        # For each wall, grab the best hole
        q_star = (q_holes + phi_shift[..., None])[jnp.arange(nwalls),best_hole]
        return q_star # Return [nwalls,]
        
    return sample_problem_params, get_problem_phi, cost, mock_solution

def main():
    import matplotlib.pyplot as plt
    nwalls = 2
    samp_prob, get_phi, cost, mock_sol = \
        get_toy_problem_functions(nwalls=nwalls)

    key = jax.random.PRNGKey(2)
    probp = samp_prob(key, batchsize=3)

    # Plot total cost for each wall, with the other wall fixed to 0
    fig, axs = plt.subplots(1,nwalls, sharey=True,figsize=(4*nwalls,4))

    q_star = mock_sol(probp)[0]
    print("q_star", q_star.shape)
    for walli in range(2):
        ax = axs[walli]
        ax.set_title(f"wall {walli}")

        if walli==0:
            qwall0 = jnp.linspace(-3., 3., 32)
            qwall1 = jnp.zeros_like(qwall0)
        else:
            qwall0 = jnp.zeros_like(qwall0)
            qwall1 = jnp.linspace(-3., 3., 32)
        qs = jnp.stack([qwall0, qwall1], axis=-1)
        c = vmap(cost, in_axes=(None,0),out_axes=(0))(qs, probp)
        cviz = c[0]
        print(c.shape)

        if walli==0:
            ax.plot(qwall0, cviz)
        else:
            ax.plot(qwall1, cviz)
        ax.axvline(q_star[walli],color='r')

    fig.savefig("./viz_toy_cost.png")

if __name__ == "__main__":
    #visualize the problem
    main()
