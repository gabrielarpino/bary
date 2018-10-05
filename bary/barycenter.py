import numpy as np
import scipy.linalg

def compute_barycenter(means, kerns, T=1000, xi=None):

    n = means[1].shape[0]

    if xi == None:
        xi = 1/len(kerns)

    for i in range(len(kerns)):
        kerns[i] = np.reshape(kerns[i], (n, n))

    # Calculate the barycenter mean
    # mm = lambda dist: np.mean(dist, x)
    mbar = sum([xi*meen for meen in means])

    # Calculate the barycenter covariance matrix`
    # Kbar = np.matmul(np.random.rand(means[1].shape[0], means[1].shape[0]), np.random.rand(means[1].shape[0], means[1].shape[0]))
    Kbar = sum([xi * kk for kk in kerns]) # Initialize Kbar to be the euclidean average
    print("KBAR:", Kbar)
    update = lambda kern, Kbar2, xi: xi * scipy.linalg.sqrtm(np.matmul(np.matmul(Kbar2, kern), Kbar2))

    for t in range(1, T):
        Kbar2 = scipy.linalg.sqrtm(Kbar)
        Kbarnew = np.sum([update(kern, Kbar2, xi) for kern in kerns], axis = 0)
        print("t, NAN IN SQRT, diff: ", t, np.isnan(Kbar2).any(), np.mean(np.abs(Kbarnew - Kbar)))
        Kbar = Kbarnew

    return mbar, Kbar


def barycenter_riccati():
    """ A faster fixed point iteration scheme that exploits the riccati structure
    of the problem """

    # I = sum(A)

    return
