import jax
import jax.numpy as jnp

nalphas = (2,2)
z_size = 2

class ZDecoder(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray


    def __init__(self, in_size, out_size, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))
        self.Z = jax.random.normal(nalphas+(z_size,))

    def __call__(self, x):
        return self.weight @ x + self.bias

@jax.jit
@jax.grad
def loss_fn(model, x, y):
    pred_y = jax.vmap(model)(x)
    return jax.numpy.mean((y - pred_y) ** 2)


q_stars_mock = jnp.array([[0.,0.], [0.,1.],[1.,0.], [1.,1.]])

def train(decoder, problem_funcs):
    # sample_problem_phi, c, mock_solution = problem_funcs
    # phi_i = sample_problem_phi()
    # q_star = mock_solution(phi_i)
    phi_i = jax.random.normal()
    q_star = (q_stars_mock + phi_i) + jax.random.normal()

    batch_size, in_size, out_size = 32, 2, 3
    model = Decoder(in_size, out_size, key=jax.random.PRNGKey(0))
    grads = loss_fn(model, x, q_star)

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        pred_y = jax.vmap(model)(x)
        # Trains with respect to binary cross-entropy
        return -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))

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
    for step, (x, y) in zip(range(steps), iter_data):
        loss, model, opt_state = make_step(model, x, y, opt_state)
        loss = loss.item()
        print(f"step={step}, loss={loss}")

    pred_ys = jax.vmap(model)(xs)
    num_correct = jnp.sum((pred_ys > 0.5) == ys)
    final_accuracy = (num_correct / dataset_size).item()
    print(f"final_accuracy={final_accuracy}")