import gpflow
import pymc3 as pm
import wbml
import numpy as np
import scipy.linalg as spa
from stheno import EQ, Delta
import matplotlib.pyplot as plt
from scipy.stats import t, cauchy
from bary import barycenter

def sample_t_dist(X, size):

    # Define the true covariance function and its parameters
    l_true = 1.0
    n_true = 3.0
    cov_func = n_true**2 * pm.gp.cov.Matern52(1, l_true)

    # A mean function that is zero everywhere
    mean_func = pm.gp.mean.Zero()

    # The latent function values are one sample from a multivariate normal
    # Note that we have to call `eval()` because PyMC3 built on top of Theano
    tp_samples = pm.MvStudentT.dist(mu=mean_func(X).eval(), cov=cov_func(X).eval(), nu=3).random(size=size)

    return tp_samples



def run():
    # Sample from a student t distribution, it should look like a gaussian but
    # with more outliers.

    # set the seed
    np.random.seed(1)

    n = 10 # The number of data points
    X = np.linspace(0, 10, n)[:, None] # The inputs to the GP, they must be arranged as a column vector

    # Define the true covariance function and its parameters
    l_true = 1.0
    n_true = 3.0
    cov_func = n_true**2 * pm.gp.cov.Matern52(1, l_true)

    # A mean function that is zero everywhere
    mean_func = pm.gp.mean.Zero()

    # The latent function values are one sample from a multivariate normal
    # Note that we have to call `eval()` because PyMC3 built on top of Theano
    tp_samples = pm.MvStudentT.dist(mu=mean_func(X).eval(), cov=cov_func(X).eval(), nu=3).random(size=8)

    ## Plot samples from TP prior
    fig = plt.figure(figsize=(12,5)); ax = fig.gca()
    ax.plot(X.flatten(), tp_samples.T, lw=3, alpha=0.6);
    ax.set_xlabel("X"); ax.set_ylabel("y"); ax.set_title("Samples from TP with DoF=3");
    plt.show()

    # Fit and optimize two GPs with the same kernels to samples from this distribution,
    # and show that the barycenter of these is more representative of the t-dist, or
    # is more robust to outliers.

    # Fit first GP to a sample
    k1 = gpflow.kernels.RBF(variance = 8.0, lengthscales=2.0, input_dim = 1)
    m1 = 0
    model1 = gpflow.models.GPR(X, tp_samples[1], k1, m1)
    model1.likelihood.variance = 0.05
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(model1)

    # Fit second GP to a sample
    k2 = gpflow.kernels.RBF(variance = 8.0, lengthscales=2.0, input_dim = 1)
    m2 = 0
    model2 = gpflow.models.GPR(X, tp_samples[2], k2, m2)
    model2.likelihood.variance = 0.05
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(model2)

    # Get barycenter of GPs
    mbar, Kbar = barycenter(m1, m2, k1(X), k2(X))

    gp_samples = pm.MvNormal.dist(mu=mean_func(X).eval(), cov=cov_func(X).eval()).random(size=8)
    fig = plt.figure(figsize=(12,5)); ax = fig.gca()
    ax.plot(X.flatten(), gp_samples.T, lw=3, alpha=0.6);
    ax.set_xlabel("X"); ax.set_ylabel("y"); ax.set_title("Samples from GP");
    plt.show()

    return

if __name__ == '__main__':
  evaluate()
