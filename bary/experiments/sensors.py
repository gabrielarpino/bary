import gpflow
import pymc3 as pm
import numpy as np
import scipy.linalg as spa
import matplotlib.pyplot as plt
import bary.barycenter

def sample_t_dist(X, size):

    # Define the true covariance function and its parameters
    l_true = 1.0
    n_true = 3.0
    cov_func = n_true**2 * pm.gp.cov.Matern52(1, l_true)

    # A mean function that is zero everywhere
    mean_func = pm.gp.mean.Zero()

    # The latent function values are one sample from a multivariate normal
    # Note that we have to call `eval()` because PyMC3 built on top of Theano
    tp_samples = pm.MvStudentT.dist(mu=mean_func(X).eval(), cov=cov_func(X).eval(), nu=3).random(size=size)

    return tp_samples

def single_exp(seed = 1):
    # Sample from a student t distribution, it should look like a gaussian but
    # with more outliers.

    # set the seed
    np.random.seed(seed)

    n = 100 # The number of data points
    n_sensors = 4
    X = np.linspace(0, 10, n)[:, None] # The inputs to the GP, they must be arranged as a column vector

    # Define the true covariance function and its parameters
    l_true = 1.0
    n_true = 3.0
    period = 0.6
    ls = 0.4
    # cov = pm.gp.cov.Periodic(1, period=period, ls=ls)
    cov_true = n_true**2 * pm.gp.cov.Matern52(1, l_true) * pm.gp.cov.Periodic(1, period=period, ls=ls)

    # Generate n_sensors number of sensors, each with an added random noise covariance on top
    cov_s1 = cov_true + 1.0 * pm.gp.cov.Matern32(1, 0.5)
    cov_s2 = cov_true + 2.0 * pm.gp.cov.Matern32(1, 1.0)
    cov_s3 = cov_true + 3.0 * pm.gp.cov.Matern32(1, 5.0)
    cov_s4 = cov_true + 4.0 * pm.gp.cov.Matern32(1, 25.0)
    covs = [c(X).eval() for c in [cov_s1, cov_s2, cov_s3, cov_s4]]

    # A mean function that is zero everywhere
    mean_true = pm.gp.mean.Zero()
    means = [mean_true(X).eval() for i in covs]

    # Now, barycenter these three covariance matrices and see if closer to cov_true than the euclidean
    sum_det = sum([np.linalg.det(c(X).eval()) for c in [cov_s1, cov_s2, cov_s3, cov_s4]])
    xi = [np.linalg.det(c(X).eval())/sum_det for c in [cov_s1, cov_s2, cov_s3, cov_s4]] # Set the xi to 1 over determinant
    mbar, Kbar = bary.barycenter.compute_barycenter(means, covs, T = 200, xi = xi)
    meuc = mbar
    Keuc = sum([xi[i]*covs[i] for i in range(len(covs))])
    Kbar = np.real(Kbar) # Sometimes complex values in there due to numerical issues

    # Evalueate predictions
    true_gp_cov = cov_true(X).eval()
    kbar_diff_norm = np.linalg.norm(true_gp_cov - Kbar) / np.linalg.norm(true_gp_cov)
    keuc_diff_norm = np.linalg.norm(true_gp_cov - Keuc) / np.linalg.norm(true_gp_cov)
    print("KBAR diff norm", kbar_diff_norm)
    print("Keuc diff norm", keuc_diff_norm)

    # Results, experiment 1, seed(1):
    # Kbar norm difference from the actual true covariance is much better than the Euclidean.
    # KBAR diff norm 0.8408399681758418
    # Keuc diff norm 0.9823041669944972
    # Now, experiment 2: Run with setting weights equal to inverse of determinant to see if both improve or just bary.
    # KBAR diff norm 1.9463826919744265
    # Keuc diff norm 1.975393199200042
    # Experiment 3: Run with setting weights equal to the determinant to see if both barycenters improve.
    # KBAR diff norm 0.1599577736927176
    # Keuc diff norm 0.1602719498265207

    return kbar_diff_norm, keuc_diff_norm

def run():
    return single_exp()

if __name__ == '__main__':
    run()
