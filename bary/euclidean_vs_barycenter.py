import gpflow
import numpy as np
import scipy
from barycenter import barycenter
import matplotlib.pyplot as plt

# Conduct an aliasing experiment: Sample at less than the nyquist frequency, and condition two
# periodic GPs on it, show which one is the real one.

# Create the data
N = 12
x = np.linspace(0, 2 * 2 * np.pi / 5, N)
x = x.reshape(x.shape[0], 1)
Yreal = np.sin(5 * x) + np.random.randn(N,1)*0.05
plt.plot(x, Yreal, 'kx', mew=2)
plt.show()

k1 = gpflow.kernels.Periodic(variance = 1.0, period = 2.5, lengthscales=1.0, input_dim = 1)
k2 = gpflow.kernels.Periodic(variance = 1.0, period = 10.0, lengthscales=1.0, input_dim = 1)
k3 = gpflow.kernels.RBF(variance = 8.0, lengthscales=2.0, input_dim = 1)
m1 = 0
m2 = 0
m3 = 0
mbar, Kbar = barycenter([m1, m2, m3], [k1.compute_K(x, x), k2.compute_K(x, x)], N=N, T=40000)
meuclid, Keuclid = (m1 + m2)/2, (k1.compute_K(x, x) + k2.compute_K(x, x))/2

# # Sample from the real distribution
# # Generate sample data from known GP
# N_predict = 80
# predict_distance = 4.0
# X = x
# X_predict = np.linspace(5.0, 5.0 + predict_distance, N_predict)
# X_full = np.vstack((X, X_predict))
# # Y = np.sin(12*X) + 0.66*np.cos(25*X) + np.random.randn(N,1)*0.01 + 3
# Y = np.sin(2 * np.pi / 1.0 * X) + np.random.randn(N,1)*0.1 + 3
# Y_predict = np.sin(2 * np.pi / 1.0 * X_predict1) + np.random.randn(N_predict,1)*0.1 + 3
# plt.plot(X, Y, 'kx', mew=2)
# plt.scatter(X_predict, Y_predict)
# plt.show()

# Condition the GPs on data
model1 = gpflow.models.GPR(x, Yreal, k1, m1)
model1.likelihood.variance = 0.05
model2 = gpflow.models.GPR(x, Yreal, k2, m2)
model2.likelihood.variance = 0.05
# model3 = gpflow.models.GPR(X, Y, k3, m3)
# model3.likelihood.variance = 0.01

def plot(m, xx):
    mean, var = m.predict_y(xx)
    plt.figure(figsize=(12, 6))
    # plt.plot(X, Y, 'kx', mew=2)
    plt.plot(xx, mean, 'C0', lw=2)
    plt.fill_between(xx[:,0],
                     mean[:,0] - 2*np.sqrt(var[:,0]),
                     mean[:,0] + 2*np.sqrt(var[:,0]),
                     color='C0', alpha=0.2)
#     plt.xlim(-0.1, 1.1)

N_pred = 40
xx = np.linspace(0, 2 * 2 * np.pi / 5, N_pred)
xx = xx.reshape(xx.shape[0], 1)
plot(model1, xx)
plt.scatter(x, Yreal)

plot(model2, xx)
plt.scatter(x, Yreal)
# plot(model3, xx)


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
