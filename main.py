import jax
from jax import random
import jax.numpy as jnp
import optax
import equinox as eqx
import pdb

from toy_problem import get_toy_problem_functions

nalphas = (2,2)
z_size = 2
learning_rate = 1e-2
steps = 100

class ZDecoder(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_size, out_size, key):
        wkey, bkey, Zkey = jax.random.split(key, num=3)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))
        # self.Z = jax.random.normal(Zkey, nalphas+(z_size,))

    def __call__(self, x):
        return jax.numpy.matmul(self.weight, jax.numpy.concatenate((x, self.bias)))


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

def train():
    samp_prob, get_phi, cost, mock_sol = get_toy_problem_functions(nwalls=2)
    key = jax.random.PRNGKey(0)
    probp = samp_prob(key, batchsize=50)
    q_star = mock_sol(probp)[0]
    phi_i = get_phi(probp)

    pdb.set_trace()

    in_size, out_size = 4, 2
    model = ZDecoder(in_size, out_size, key=jax.random.PRNGKey(0))

    @eqx.filter_value_and_grad 
    def compute_loss(model, x, y):
        pred_y = model(x) #jax.vmap(model)(x)
        return jax.numpy.mean((y - pred_y) ** 2)

    # Important for efficiency whenever you use JAX: wrap everything into a single JIT
    # region.
    @eqx.filter_jit
    def make_step(model, x, y, opt_state):
        loss, grads = compute_loss(model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)
    for epoch in range(500):
        losses = []
        for sample in range(q_stars_mock.shape[0]):
            loss, model, opt_state = make_step(model, phi_i[sample, :], q_star[sample, :], opt_state)
            loss = loss.item()
            losses.append(loss)
        print(f"epoch={epoch}, loss={sum(losses)/len(losses)}")

if __name__=='__main__':
    train()