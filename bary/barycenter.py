import numpy as np

def barycenter(means, kerns, N = 100, T=10000, xi=None):

    x = np.linspace(0., 1., N)

    if xi == None:
        xi = 1/len(kerns)

    # Calculate the barycenter mean
    # mm = lambda dist: np.mean(dist, x)
    mbar = sum([xi*meen for meen in means])

    # Calculate the barycenter covariance matrix`
    Kbar = np.random.rand(N, N)
    update = lambda kern, Kbar2, xi: xi * np.sqrt(Kbar2 * kern * Kbar2)

    for t in range(1, T):
        Kbar2 = np.sqrt(Kbar)
        Kbar = np.sum([update(kern, Kbar2, xi) for kern in kerns], axis = 0)

    return mbar, Kbar


def barycenter_riccati():
    """ A faster fixed point iteration scheme that exploits the riccati structure
    of the problem """

    # I = sum(A)

    return
