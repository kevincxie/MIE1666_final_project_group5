import jax.numpy as jnp
from jax import vmap
from functools import partial

def get_toy_problem_functions():
    def sample_problem_phi():
        """
        Returns:
            phi: 
        """
        return phi

    @partial(jnp.vectorize, signature='(k)->()')
    def c(q, phi):
        """ Batched cost function computation """
        # assert x.ndim == 2
        # assert x.shape[1] == 2
        A = jnp.array([[1.,0.2],[-0.3,1.]])
        # return jnp.einsum('bi,bi->b', x, jnp.einsum('ij,bj->bi',A, x))
        return jnp.vdot(x,jnp.matmul(A,x))
    
    def mock_solution(phi):
        """
        Pretends to do VOO
        Return:
            q_star: Approximate solution to phi    
        """
        return

    return sample_problem_phi, c, mock_solution
