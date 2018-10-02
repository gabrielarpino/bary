import numpy as np
import scipy.linalg as spa
from stheno import EQ, Delta
import matplotlib.pyplot as plt

def random_K(n):
    Ki = np.random.randn(n, n)
    return Ki.T @ Ki


def root(a):
    u, s_diag, _ = np.linalg.svd(a)
    return u.dot(np.diag(s_diag ** .5)).dot(u.T)


def fp_difference(K, *Ks):
    root_K = root(K)
    return K - np.mean([root(root_K @ Ki @ root_K) for Ki in Ks], axis=0)


from bary import barycenter
import gpflow
mbar, Kbar = barycenter([0., 0., 0.], [K1, K2, K3], N=n, T=400)

# Saw that the optimal transport thing reduces to the ricatti equation
n = 20
x = np.linspace(0, 5, n)
x = x.reshape(x.shape[0], 1)
K1 = random_K(n)
K2 = random_K(n)
K3 = random_K(n)

# Solve.
print('Computing...')
C1 = spa.solve_continuous_are(
    np.zeros((n, n)), np.linalg.cholesky(K1), K3, np.eye(n))
print('Symdiff C1:', np.sum(np.abs(C1 - C1.T)))
print('Mineig C1:', np.min(np.real(np.linalg.eig(C1)[0])))
B1 = spa.solve_continuous_are(
    np.zeros((n, n)), np.linalg.cholesky(K2), K1, np.eye(n))
print('Symdiff B1:', np.sum(np.abs(B1 - B1.T)))
print('Mineig B2:', np.min(np.real(np.linalg.eig(B1)[0])))
iA1 = (np.eye(n) + C1 + np.linalg.inv(B1)) / 3
print('Symdiff iA1:', np.sum(np.abs(iA1 - iA1.T)))
print('Mineig iA1:', np.min(np.real(np.linalg.eig(iA1)[0])))

K_from1 = iA1 @ K1 @ iA1
K = K_from1

# Check difference.
print('Checking...')
print('L1 norm of FP difference:',
      np.sum(np.abs(fp_difference(K, K1, K2, K3))))

# Plot results.
plt.plot(x, K1[0, :], label='$k_1$', ls='dashed')
plt.plot(x, K2[0, :], label='$k_2$', ls='dashed')
plt.plot(x, K3[0, :], label='$k_2$', ls='dashed')
plt.plot(x, K[0, :], label='Barycenter')
plt.plot(x, (K1[0, :] + K2[0, :]) / 2, label='Euclidean average')
plt.xlabel('$t$')
plt.legend()
plt.show()
