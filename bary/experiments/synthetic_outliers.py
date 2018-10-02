import gpflow
import pymc3 as pm
import wbml
import numpy as np
import scipy.linalg as spa
from stheno import EQ, Delta
import matplotlib.pyplot as plt
from scipy.stats import t, cauchy

# Sample from a student t distribution, it should look like a gaussian but
# with more outliers.

# set the seed
np.random.seed(1)

n = 100 # The number of data points
X = np.linspace(0, 10, n)[:, None] # The inputs to the GP, they must be arranged as a column vector

# Define the true covariance function and its parameters
ℓ_true = 1.0
η_true = 3.0
cov_func = η_true**2 * pm.gp.cov.Matern52(1, ℓ_true)

# A mean function that is zero everywhere
mean_func = pm.gp.mean.Zero()

# The latent function values are one sample from a multivariate normal
# Note that we have to call `eval()` because PyMC3 built on top of Theano
tp_samples = pm.MvStudentT.dist(mu=mean_func(X).eval(), cov=cov_func(X).eval(), nu=3).random(size=8)

## Plot samples from TP prior
fig = plt.figure(figsize=(12,5)); ax = fig.gca()
ax.plot(X.flatten(), tp_samples.T, lw=3, alpha=0.6);
ax.set_xlabel("X"); ax.set_ylabel("y"); ax.set_title("Samples from TP with DoF=3");


# Fit and optimize two GPs with the same kernels to samples from this distribution,
# and show that the barycenter of these is more representative of the t-dist, or
# is more robust to outliers.

gp_samples = pm.MvNormal.dist(mu=mean_func(X).eval(), cov=cov_func(X).eval()).random(size=8)
fig = plt.figure(figsize=(12,5)); ax = fig.gca()
ax.plot(X.flatten(), gp_samples.T, lw=3, alpha=0.6);
ax.set_xlabel("X"); ax.set_ylabel("y"); ax.set_title("Samples from GP");
