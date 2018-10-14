import gpflow
import pymc3 as pm
import numpy as np
import scipy.linalg as spa
import matplotlib.pyplot as plt
import bary.barycenter as b

n = 10 # The number of data points
X = np.linspace(0, 10, n)[:, None]

cov_1 = 1**2 * pm.gp.cov.Matern52(1, 1) * pm.gp.cov.Periodic(1, period=2, ls=2)
cov_2 = 2.0**2 * pm.gp.cov.Matern32(1, 5.0)

covs = [c(X).eval() for c in [cov_1, cov_2]]
means = [0, 0]
T = 100

print("Optmizing weights!")
res = b.optimize_weights(means, covs, T)
print(res)

# Results:

# Minimizing and maximizing the entropy of the barycenter distribution by cahnging the weights
# yields the trivial result of the weights either fully favouring the lowest or highest entropy distribuion.
# Optimizing while minimizing the KL between the two distributions, however, yields a non-trivial result.
# One of the optimized weights being ~ 0.55 with a vanishing gradient.
