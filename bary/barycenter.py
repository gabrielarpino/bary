import numpy as np
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


def barycenter_riccati():
    """ A faster fixed point iteration scheme that exploits the riccati structure
    of the problem """

    # I = sum(A)

    return
