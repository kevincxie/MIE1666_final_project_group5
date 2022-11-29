import jax
import jax.numpy as jnp
import numpy as np
import pdb
from main import ZDecoder
import equinox as eqx
import matplotlib.pyplot as plt
import os

class RBF():
    def __init__(self, sigma=None):
        self.sigma = sigma

    def __call__(self, qs):        
        n_particles = qs.shape[0]
        XY = jnp.matmul(qs, jnp.transpose(qs))
        XX = jnp.tile(jnp.diag(XY), (n_particles, 1))
        YY = jnp.transpose(XX)

        dnorm2 = -2 * XY + XX + YY

        # Apply the median heuristic 
        if self.sigma is None:
            np_dnorm2 = dnorm2
            h = jnp.median(np_dnorm2) / (2 * jnp.log(n_particles + 1))
            sigma = np.sqrt(h)
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        
        K_XY = jnp.exp(- gamma * dnorm2)
        
        dK_XY = jnp.zeros(qs.shape)

        for i in range(n_particles):
            dK_XY = dK_XY.at[i].set(jnp.matmul(K_XY[i], qs[i] - qs) * 2 * gamma)

        pdb.set_trace()
     
        return K_XY, dK_XY   

class SVGD():
    def __init__(self, qs, num_particles, seed):
        key = jax.random.PRNGKey(seed)
        t = qs.shape
        t = tuple(((num_particles-t[0])*t[1], t[2]))

        self.qs = jnp.vstack((jnp.reshape(qs, (qs.shape[0]*qs.shape[1], qs.shape[2])), jax.random.normal(key, t)))
        self.K = RBF()
        self.num_particles = self.qs.shape[0]

    def score(self, gt):
        return compute_loss(self.qs, gt)

    def phi(self, gt, svgd_r):
        logp, dlogp = self.score(gt)
        K_XX, grad_K = self.K(self.qs)
        phi = (jnp.matmul(K_XX, dlogp) + svgd_r * grad_K) / self.num_particles
        return -phi, logp.sum() / self.num_particles

    def step(self, gt, svgd_r):
        phi, cost = self.phi(gt, svgd_r)
        self.qs += phi
        return cost

    def mode_centers(self, gt):
        gt = jnp.tile(gt, (self.num_particles, 1, 1)) # num_particles x gt.shape
        dist = jnp.abs(gt - jnp.expand_dims(self.qs, 1)).sum(axis=-1)
        idx = jnp.argmin(dist, axis=0)
        return self.qs[idx]

    def optimize(self, epochs, gt, svgd_r):
        for i in range(epochs):
            cost = self.step(gt, svgd_r)
            if i%10 == 0:
                print(f"SVGD step={i+1}, loss={cost : .4f}")
        best_q = self.mode_centers(gt)
        return best_q

@eqx.filter_value_and_grad
def compute_loss(qs, q_star):
    q_fit = (jnp.expand_dims(qs, 1) - jnp.tile(q_star, (qs.shape[0], 1 ,1)))**2
    loss = jnp.sum(q_fit) / q_fit.shape[0]
    return loss

def plot_distr(qs_init, qs_fin, q_star):

    ngrid = 100
    x = np.linspace(-2, 2, ngrid)
    y = np.linspace(-2, 2, ngrid)
    X, Y = jnp.meshgrid(x,y)
        
    xy = jnp.stack((jnp.reshape(X, X.shape[0]**2), jnp.reshape(Y, Y.shape[0]**2)), axis=1)
    loss = jnp.reshape(jnp.sum((xy - q_star)**2, axis=1), (X.shape[0], Y.shape[0]))
    prob = jnp.exp(-loss) 

    # plt.plot(x_all[:, 0], x_all[:, 1], 'ro', markersize=5)

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.contourf(X, Y, prob, 30)
    ax2.contourf(X, Y, prob, 30)
    ax1.scatter(qs_init[:, 0], qs_init[:, 1])
    ax2.scatter(qs_fin[:, 0], qs_fin[:, 1])
    ax1.set_xlim([-2,2])
    ax1.set_ylim([-2,2])
    ax2.set_xlim([-2,2])
    ax2.set_ylim([-2,2])
   
    fig.savefig(os.path.join(os.path.dirname(__file__), 'svgd_optim.png'))


def main():

    nlevels = 2
    nparticles = 50
    key = jax.random.PRNGKey(0)
    latent_dim = 2
    regions = 4
    iters = 100

    # Initialize particles
    qs = jax.random.normal(key, (nparticles, nlevels))
    
    # Initialize model
    # phi_size = nlevels
    # in_size, out_size = latent_dim + phi_size, nlevels
    # model = ZDecoder(nlevels, regions, latent_dim, in_size, out_size, key=jax.random.PRNGKey(0))

    # Sample problem
    samp_prob, get_phi, cost, mock_sol = get_toy_problem_functions(nwalls=nlevels)
    psi = samp_prob(key, batchsize=1)
    gt = mock_sol(psi)

    # SVGD
    kernel = RBF()
    svgd = SVGD(kernel)
    qs_init = qs
    for i in range(iters):
        qs, cost = svgd.step(qs, gt, compute_loss, svgd_r=1)
        print("Iter {}: loss {}".format(i, cost))
    qs_fin = qs

    plot_distr(qs_init, qs_fin, gt)
    


if __name__ == '__main__':
    main()