import gpflow
import numpy as np
import scipy
from barycenter import barycenter
import matplotlib.pyplot as plt

# Saw that the optimal transport thing reduces to the ricatti equation
N = 200
x = np.linspace(0, 5, N)
x = x.reshape(x.shape[0], 1)
k1 = 3 * gpflow.kernels.Periodic(period = 1.0, lengthscales=1.2, input_dim = 1).compute_K(x, x)
k2 = 8 * gpflow.kernels.RBF(lengthscales=2.0, input_dim = 1).compute_K(x, x)
k3 = 3 * gpflow.kernels.Periodic(period = 0.1, lengthscales=0.2, input_dim = 1).compute_K(x, x)
m1 = 0
m2 = 0
m3 = 0
mbar, Kbar = barycenter([m1, m2, m3], [k1, k2, k3], N=N, T=40000)
meuclid, Keuclid = (m1 + m2 + m3)/3, (k1 + k2 + k3)/3

# # Get bary of first 2
# mbar1, Kbar1 = barycenter([m1, m2], [k1, k2], N=N, T=40000, xi = 1/3)
# # meuclid, Keuclid = (m1 + m2)/3, (k1 + k2)/3
# # Get bary with third
# mbar2, Kbar2 = barycenter([mbar1, m3], [Kbar1, k3], N=N, T=40000, xi = 1/3)
#
# # Get bary without inserting xi
# mbar1noxi, Kbar1noxi = barycenter([m1, m2], [k1, k2], N=N, T=40000)
# # meuclid, Keuclid = (m1 + m2)/3, (k1 + k2)/3
# # Get bary with third
# mbar2noxi, Kbar2noxi = barycenter([mbar1noxi, m3], [Kbar1noxi, k3], N=N, T=40000)

plt.plot(Kbar[:, 0], label="Full Wasserstein Barycenter")
# plt.plot(Kbar2[:, 0], label="Partial Wasserstein Barycenter")
# plt.plot(Kbar2noxi[:, 0], label="Partial noxi Wasserstein Barycenter")
plt.plot(Keuclid[:, 0], label="Euclid Barycenter")
plt.plot(k1[:, 0], label='K1')
plt.plot(k2[:, 0], label='K2')
plt.plot(k3[:, 0], label='K3')
plt.legend()
plt.show()

# # Solve the Ricatti equation
# # (A’X + XA - XBR^-1B’X+Q=0)
# A_ = np.zeros((N, N))
# # R_ = np.linalg.ing(k1)
# R_ = np.eye(N)
# B_ = np.linalg.cholesky(k1)
# Q_ = k2
# C = scipy.linalg.solve_continuous_are(A_, B_, Q_, R_)
#
# # Now solve for A
# A = np.linalg.inv((C + np.eye(N)) / 2)
#
# # Now solve for K_ot
# A_inv = (C + np.eye(N)) / 2
# K_ric = np.matmul(np.matmul(A_inv, k1), A_inv)
#
# # Test that it works
# np.testing.assert_almost_equal(K_ric, 1/2 * np.sqrt(np.sqrt(K_ric) * k1 * np.sqrt(K_ric)) + 1/2 * np.sqrt(np.sqrt(K_ric) * k2 * np.sqrt(K_ric)))
#
# print("Diff:", np.mean(np.abs(K_ric - Kbar)))
#
# # Alternate ricatti formulation (expanded)
# Q_ = -k1 - k2
# A_ = -2*k1
# B_ = np.eye(N)
# R_ = np.linalg.inv(-4 * k1)
# A_inv = scipy.linalg.solve_continuous_are(A_, B_, Q_, R_)
# K_ric = A_inv * k1 * A_inv
# np.testing.assert_almost_equal(K_ric, 1/2 * np.sqrt(np.sqrt(K_ric) * k1 * np.sqrt(K_ric)) + 1/2 * np.sqrt(np.sqrt(K_ric) * k2 * np.sqrt(K_ric)))
