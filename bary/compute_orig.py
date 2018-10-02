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


t_max = 5
n = 400

# k1 = EQ().stretch(t_max / 2) * EQ().periodic(1) + 1e-6 * Delta()
# k2 = EQ().stretch(t_max / 2) * EQ().periodic(1.8) + 1e-6 * Delta()
k1 = EQ().periodic(0.8) + 1e-6 * Delta()
k2 = EQ().periodic(0.85) + 1e-6 * Delta()
k3 = EQ().periodic(0.9) + 1e-6 * Delta()

x = np.linspace(0, t_max, n)

K1 = random_K(n)
K2 = random_K(n)
K3 = random_K(n)

# K1, K2, K3 = random_K(n), random_K(n), random_K(n)

# Solve.
print('Computing...')
C1 = spa.solve_continuous_are(
    np.zeros((n, n)), np.linalg.cholesky(K1), K3, np.eye(n))
print('Symdiff C1:', np.sum(np.abs(C1 - C1.T)))
print('Mineig C2:', np.min(np.real(np.linalg.eig(C1)[0])))
C2 = spa.solve_continuous_are(
    np.zeros((n, n)), np.linalg.cholesky(K2), K3, np.eye(n))
print('Symdiff C2:', np.sum(np.abs(C2 - C2.T)))
print('Mineig C2:', np.min(np.real(np.linalg.eig(C2)[0])))
iA1 = (np.eye(n) + C1 + np.linalg.solve(C2, C1)) / 3.0
iA2 = (np.eye(n) + C2 + np.linalg.solve(C1, C2)) / 3.0
print('Symdiff iA1:', np.sum(np.abs(iA1 - iA1.T)))
print('Symdiff iA2:', np.sum(np.abs(iA2 - iA2.T)))

# # Check that they satisfy the right thing.
# A1 = np.linalg.inv(iA1)
# A2 = np.linalg.inv(iA2)
# print(C1 @ A1 + A1 + A2)
# print(C2 @ A2 + A2 + A1)

K_from1 = iA1 @ K1 @ iA1.T
K_from2 = iA2 @ K2 @ iA2.T
print('Inconsistency:', np.sum(np.abs(K_from1 - K_from2)))
K = K_from2

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
