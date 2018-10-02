import numpy as np
import scipy.linalg as spa
from stheno import EQ, Delta
import gpflow
import matplotlib.pyplot as plt

def plot(m, K, x):
    var = np.diag(K).reshape(K.shape[0], 1)
    print(var.shape)
    print(var)
    plt.plot(x, m, 'C0', lw=2)
    plt.fill_between(x[:,0],
                     m[:,0] - 2*np.sqrt(var[:,0]),
                     m[:,0] + 2*np.sqrt(var[:,0]),
                     color='C0', alpha=0.2)

def random_K(n):
    Ki = np.random.randn(n, n)
    return Ki.T @ Ki


def root(a):
    u, s_diag, _ = np.linalg.svd(a)
    return u.dot(np.diag(s_diag ** .5)).dot(u.T)


def fp_difference(K, *Ks):
    root_K = root(K)
    return K - np.mean([root(root_K @ Ki @ root_K) for Ki in Ks], axis=0)

# Sample from a true GP
t_max = 2
n = 30
X = np.linspace(0, t_max, n).reshape(n, 1)
x_samp = np.random.rand(1, 1)
y_samp = np.random.rand(1, 1)
k_true = gpflow.kernels.RBF(variance = 1.0, lengthscales=1.0, input_dim = 1)
mean_true = gpflow.mean_functions.Linear(0,0)
m_true = gpflow.models.GPR(x_samp, y_samp, k_true, mean_true)
m_true.likelihood.variance = 0.1
N_samp = 1
samps1 = m_true.predict_f_samples(X, N_samp).reshape(N_samp, len(X))
y_obs = samps1[0].reshape(len(samps1[0]), 1)
x_obs = X

# Train one well specified model, and one non specified one. Get the barycenter and compare with euclidean avg.
k1 = gpflow.kernels.RBF(variance = 2.0, lengthscales=2.0, input_dim = 1)
mean1 = gpflow.mean_functions.Linear(0,0)
m1 = gpflow.models.GPR(x_obs, y_obs, k1, mean1)
m1.likelihood.variance = 0.05
# y_obs = np.sin(12*x_obs) + 0.66*np.cos(24*x_obs) + np.random.randn(n,1)*0.01
opt = gpflow.train.ScipyOptimizer()
opt.minimize(m1)

# Not well specified model (Exp kernel)
k2 = gpflow.kernels.Matern12(variance = 2.0, lengthscales=2.0, input_dim = 1)
mean2 = gpflow.mean_functions.Linear(0,0)
m2 = gpflow.models.GPR(x_obs, y_obs, k2, mean2)
m2.likelihood.variance = 0.05
# y_obs = np.sin(12*x_obs) + 0.66*np.cos(24*x_obs) + np.random.randn(n,1)*0.01
opt = gpflow.train.ScipyOptimizer()
opt.minimize(m2)

# Prepare for computation
Mean1, K1 = m1.predict_f_full_cov(x_obs)
Mean2, K2 = m2.predict_f_full_cov(x_obs)
K1 = K1.reshape(n, n) + 1e-14 * np.eye(n)
K2 = K2.reshape(n, n) + 1e-14 * np.eye(n)
Mean1 = Mean1.reshape(n, 1)
Mean2 = Mean2.reshape(n, 1)
# K1 = k1.compute_K(x_obs, x_obs) + 1e-14*np.eye(n)
# K2 = k2.compute_K(x_obs, x_obs) + 1e-14*np.eye(n)
# Mean1 = np.array([0 for i in x_obs]).reshape(n, 1)
# Mean2 = np.array([0 for i in x_obs]).reshape(n, 1)

# Find the euclidean barycenter
meuclid, Keuclid = (Mean1 + Mean2)/2, (K1 + K2)/2

# Find the wasserstein barycenter
# Solve.
print('Computing...')
C1 = spa.solve_continuous_are(
    np.zeros((n, n)), np.linalg.cholesky(K1), K2, np.eye(n))
print('Symdiff C1:', np.sum(np.abs(C1 - C1.T)))
print('Mineig C1:', np.min(np.real(np.linalg.eig(C1)[0])))
iA1 = (np.eye(n) + C1) / 2.0
K_from1 = iA1 @ K1 @ iA1.T
K = K_from1
print('Checking...')
print('L1 norm of FP difference:',
      np.sum(np.abs(fp_difference(K, K1, K2))))
mw, Kw = (Mean1 + Mean2)/2, K

# Compare the two barycenters
plot(Mean2, K2, x_obs)
plot(Mean1, K1, x_obs)
plot(mw, Kw, x_obs)
# plot(mw, Kw, x_obs)


# Plot results.
plt.plot(x, K1[0, :], label='$k_1$', ls='dashed')
plt.plot(x, K2[0, :], label='$k_2$', ls='dashed')
plt.plot(x, K3[0, :], label='$k_2$', ls='dashed')
plt.plot(x, K[0, :], label='Barycenter')
plt.plot(x, (K1[0, :] + K2[0, :]) / 2, label='Euclidean average')
plt.xlabel('$t$')
plt.legend()
plt.show()
