import jax
from jax import random
import jax.numpy as jnp
import optax
import equinox as eqx
import pdb

from argparse import ArgumentParser

from toy_problem import get_toy_problem_functions

learning_rate = 1e-2
steps = 100

class ZDecoder(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray
    region_params: jnp.ndarray

    z_size: int
    levels: int
    regions: int

    def __init__(self, args, in_size, out_size, key):

        assert(args.levels <= args.latent_dim and args.latent_dim % args.levels == 0)

        self.z_size = args.latent_dim // args.levels
        self.levels = args.levels
        self.regions = args.regions

        wkey, bkey, Zkey = jax.random.split(key, num=3)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))
        self.region_params = jax.random.normal(Zkey, (args.levels, args.regions, self.z_size))


    def __call__(self, phi):
        batch_size, dim = phi.shape

        regions, levels, z_size = self.regions, self.levels, self.z_size

        # cartesian product
      # Can't do mesh grid on n-d and no cartesian product :(
      # grids = jnp.meshgrid(*self.Z)
      # V_alphas = jnp.stack(grids, axis=-1)
      # V_alphas = V_alphas.reshape(-1, z_size) # [region count, z_dim]
      #
        # stuff that i'm not even sure is actually any faster than using itertools.product
        i = jnp.mgrid[(slice(0, regions, 1),)*levels]
        i = jnp.stack(i, axis=-1).reshape(-1, levels)
        V_alphas = self.region_params[i[:, jnp.arange(levels)], jnp.arange(levels)].reshape(-1, levels*z_size)

        phi = jnp.broadcast_to(phi.reshape(batch_size, 1, dim), (batch_size, V_alphas.shape[0], dim))
        V_alphas = jnp.broadcast_to(V_alphas, (batch_size,)+V_alphas.shape)

        X = jnp.concatenate([V_alphas, phi], axis=-1) # batch, nregions*nlevels, z_size+phi_dim
        qs = jnp.matmul(X, self.weight.T) #[batch, nregions*nlevels, q_size]

        return qs



q_stars_mock = jnp.array([[0.,0.], [0.,1.],[1.,0.], [1.,1.]])


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



def train(args):
    key = jax.random.PRNGKey(args.seed)

    samp_prob, get_phi, cost, mock_sol = get_toy_problem_functions(nwalls=args.prob_dim)

    probp = samp_prob(key, batchsize=50)
    q_star = mock_sol(probp)
    print(q_star.shape)
    phi_i = get_phi(probp)
    print(phi_i.shape)

    #pdb.set_trace()
    phi_size = args.prob_dim

    in_size, out_size = args.latent_dim + phi_size, args.prob_dim
    model = ZDecoder(args, in_size, out_size, key=jax.random.PRNGKey(0))

    @eqx.filter_value_and_grad 
    def compute_loss(model, phi, q_star):
        qs = model(phi)

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

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)
    for epoch in range(args.epochs):
        #for sample in range(q_stars_mock.shape[0]):
        probp = samp_prob(key, batchsize=args.problem_batch_size)
        phi = get_phi(probp)
        q_star = mock_sol(probp)
        loss, model, opt_state = make_step(model, phi, q_star, opt_state)

        loss = loss.item()
        #losses.append(loss)
        print(f"epoch={epoch}, loss={loss}")


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--problem_batch_size", type=int, default=50, help="For each iteration how many problems to sample")
    parser.add_argument("--epochs", type=int, default=100, help="Total iteration count")

    parser.add_argument("--levels", type=int, default=2, help="Number of levels to the problem, can't be greater than latent dimension")
    parser.add_argument("--regions", type=int, default=2, help="Number of voronoi regions per level")

    parser.add_argument("--latent_dim", type=int, default=2, help="Dimensions in the latent space")
    parser.add_argument("--prob_dim", type=int, default=2, help="Number of dimensions in the problem, corresponds to the number of walls")

    args = parser.parse_args()

    train(args)
