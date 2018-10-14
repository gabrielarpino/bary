from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam
import autograd.scipy.linalg as ss
import scipy.linalg

def compute_barycenter(means, kerns, T=1000, xi=None):

    n = means[1].shape[0]

    if xi == None:
        xi = [1/len(kerns) for i in means]

    for i in range(len(kerns)):
        kerns[i] = np.reshape(kerns[i], (n, n))

    # Calculate the barycenter mean
    # mm = lambda dist: np.mean(dist, x)
    mbar = sum([xi[i]*means[i] for i in range(len(means))])

    # Calculate the barycenter covariance matrix`
    # Kbar = np.matmul(np.random.rand(means[1].shape[0], means[1].shape[0]), np.random.rand(means[1].shape[0], means[1].shape[0]))
    Kbar = sum([xi[i] * kerns[i] for i in range(len(kerns))]) # Initialize Kbar to be the euclidean average
    print("KBAR:", Kbar)
    update = lambda kern, Kbar2, x_i: x_i * scipy.linalg.sqrtm(np.matmul(np.matmul(Kbar2, kern), Kbar2))

    for t in range(1, T):
        Kbar2 = scipy.linalg.sqrtm(Kbar)
        Kbarnew = np.sum([update(kerns[i], Kbar2, xi[i]) for i in range(len(kerns))], axis = 0)
        print("t, NAN IN SQRT, diff: ", t, np.isnan(Kbar2).any(), np.mean(np.abs(Kbarnew - Kbar)))
        Kbar = Kbarnew

    return mbar, Kbar

def barycenter_alpha(means, kerns, T, xi):
    """ The barycenter function used for optimization"""

    def sig(x): return 1 / (1 + np.exp(-1 * x))

    n = kerns[0].shape[0]

    for i in range(len(kerns)):
        kerns[i] = np.reshape(kerns[i], (n, n))

    # Calculate the barycenter mean
    # mm = lambda dist: np.mean(dist, x)
    # mbar = sum([sig(xi)*means[i] for i in range(len(means))]) # just two dists for now
    # mbar = sig(xi)*means[0] + (1-sig(xi))*means[1]
    mbar = 0

    # Calculate the barycenter covariance matrix`
    # Kbar = np.matmul(np.random.rand(means[1].shape[0], means[1].shape[0]), np.random.rand(means[1].shape[0], means[1].shape[0]))
    # Kbar = sum([xi[i] * kerns[i] for i in range(len(kerns))]) # Initialize Kbar to be the euclidean average
    Kbar = sig(xi) * kerns[0] + (1-sig(xi)) * kerns[1] # Initialize Kbar to be the euclidean average
    # print("KBAR:", Kbar)
    update = lambda kern, Kbar2, x_i: x_i * ss.sqrtm(np.matmul(np.matmul(Kbar2, kern), Kbar2))

    for t in range(1, T):
        Kbar2 = ss.sqrtm(Kbar)
        Kbarnew = update(kerns[0], Kbar2, sig(xi)) + update(kerns[1], Kbar2, (1-sig(xi)))
        # print("t, NAN IN SQRT, diff: ", t, np.isnan(Kbar2).any(), np.mean(np.abs(Kbarnew - Kbar)))
        Kbar = Kbarnew

    return mbar, Kbar

def optimize_weights(means, kerns, T):

    def KL(K1, K2):
        return np.log(np.linalg.det(K2) / np.linalg.det(K1)) + np.trace(np.matmul(np.linalg.inv(K2), K1))

    def H(K):
        return np.log(np.linalg.det(K))

    def objective1(xi, iter):
        """ Maximize of minimize the entropy"""
        mbar, Kbar = barycenter_alpha(means, kerns, T, xi)
        return H(Kbar)

    def objective2(xi, iter):
        """ Maximize or minimize the KL divergence between the Kbar and the original K_i's """
        mbar, Kbar = barycenter_alpha(means, kerns, T, xi)
        return KL(kerns[0], Kbar) + KL(kerns[1], Kbar)

    # Get gradient of objective using autograd.
    objective_grad = grad(objective2)
    print("Running the objective!")
    print(objective2(0.0, 1))
    print("RUNNING OBJECTIVE GRAD")
    print(objective_grad(0.0, 1))

    print("     Params     |    Iter  |      Gradient  ")
    def print_perf(params, iter, gradient, H):
        print("{:15}|{:20}|{:20}|{:20}".format(params, iter, gradient, H))

    xi = 0.0
    lr = 0.01
    for i in range(1, 500):
        g = objective_grad(xi, i)
        xi -= lr * g
        if i%10==0:
            print_perf(xi, i, g, objective2(xi, i))

    # # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    # optimized_params = adam(objective_grad, [1.0], step_size=0.01,
    #                         num_iters=100, callback=print_perf)


    return xi

def barycenter_riccati():
    """ A faster fixed point iteration scheme that exploits the riccati structure
    of the problem """

    # I = sum(A)

    return
