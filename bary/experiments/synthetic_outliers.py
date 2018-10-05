import gpflow
import pymc3 as pm
import wbml
import numpy as np
import scipy.linalg as spa
from stheno import EQ, Delta
import matplotlib.pyplot as plt
from scipy.stats import t, cauchy
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

def single_exp(num_samples = 12, outlier_mag = 0.1, seed = 1):
    # Sample from a student t distribution, it should look like a gaussian but
    # with more outliers.

    # set the seed
    np.random.seed(seed)

    n = 100 # The number of data points
    X = np.linspace(0, 10, n)[:, None] # The inputs to the GP, they must be arranged as a column vector

    # Define the true covariance function and its parameters
    l_true = 1.0
    n_true = 3.0
    period = 0.6
    ls = 0.4
    # cov = pm.gp.cov.Periodic(1, period=period, ls=ls)
    cov_func = n_true**2 * pm.gp.cov.Matern52(1, l_true) * pm.gp.cov.Periodic(1, period=period, ls=ls)

    # A mean function that is zero everywhere
    mean_func = pm.gp.mean.Zero()

    # The latent function values are one sample from a multivariate normal
    # Note that we have to call `eval()` because PyMC3 built on top of Theano
    # tp_samples = pm.MvStudentT.dist(mu=mean_func(X).eval(), cov=cov_func(X).eval(), nu=3).random(size=8)
    # print("SHAPE:", tp_samples.shape)

    # ## Plot samples from TP prior
    # fig = plt.figure(figsize=(12,5)); ax = fig.gca()
    # ax.plot(X.flatten(), tp_samples.T, lw=3, alpha=0.6);
    # ax.set_xlabel("X"); ax.set_ylabel("y"); ax.set_title("Samples from TP with DoF=3");
    # plt.show()

    # num_samples = 12
    gp_samples = pm.MvNormal.dist(mu=mean_func(X).eval(), cov=cov_func(X).eval()).random(size=num_samples)
    # fig = plt.figure(figsize=(12,5)); ax = fig.gca()
    # ax.plot(X.flatten(), gp_samples.T, lw=3, alpha=0.6);
    # ax.set_xlabel("X"); ax.set_ylabel("y"); ax.set_title("Samples from GP");
    # plt.show()

    # Explicitly add outliers to the first gp sample
    def noisy_sample(eps, i): return gp_samples[i] + eps * np.random.standard_cauchy(size=gp_samples[1].shape[0])
    # fig = plt.figure(figsize=(12,5)); ax = fig.gca()
    # ax.plot(X.flatten(), np.vstack((noisy_sample(0.1, 1), gp_samples[1])).T, lw=3, alpha=0.6);
    # ax.set_xlabel("X"); ax.set_ylabel("y"); ax.set_title("Cauchy Modified Sample");
    # plt.show()

    # Fit and optimize 3 GPs, each with a certain value of cauchy noise in the sample.
    # Then, get the barycenter of these, and the wasserstein barycenter should favour the ones with lower noise more, and approximate the true gp sample.

    # Now, sample 8 times from the high dimensional gaussian, fit 8 different gps to the samples, and get the barycenters of these

    ks = []
    ms = []
    samps = []
    models = []
    opts = []
    means = []
    covs = []
    for i in range(num_samples):
        # Fit first GP to a sample
        k1 = gpflow.kernels.Matern52(variance = 1.0, lengthscales = 2.0, input_dim = 1) * gpflow.kernels.Periodic(variance = 1.0, period=1.0, lengthscales=1.0, input_dim = 1)
        m1 = 0
        samp1 = np.reshape(noisy_sample(outlier_mag, i), (n, 1))
        model1 = gpflow.models.GPR(X, samp1, k1, m1)
        model1.likelihood.variance = 0.1
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(model1)
        mean1, cov1 = model1.predict_f_full_cov(X)

        # Save them in lists
        ks.append(k1)
        ms.append(m1)
        samps.append(samp1)
        models.append(model1)
        opts.append(opt)
        means.append(mean1)
        covs.append(cov1)

    # Get wasserstein barycenter of GPs
    mbar, Kbar = bary.barycenter.compute_barycenter(means, covs, T = 100)

    # Get the euclidean barycenter of GPs
    meuc = mbar
    Keuc = 1/(len(covs)) * sum([i for i in covs])
    Kbar = np.real(Kbar) # Sometimes complex values in there due to numerical issues

    # # # Now that have the wasserstein barycenter, plot it
    # plt.figure(figsize=(12, 6))
    # # plt.plot(X, Y, 'kx', mew=2)
    # plt.plot(X, mbar, 'C0', lw=2)
    # a = np.diag(Kbar)
    # sa = np.reshape(np.sqrt(a), (n, 1))
    # plt.fill_between(X[:, 0],
    #                  mbar[:, 0] - 2*sa[:, 0],
    #                  mbar[:, 0] + 2*sa[:, 0],
    #                  color='C0', alpha=0.2)
    #
    # # Plot the euclidean barycenter!
    # plt.plot(X, meuc, 'C1', lw=2)
    # a = np.diag(Keuc)
    # sa = np.reshape(np.sqrt(a), (n, 1))
    # plt.fill_between(X[:, 0],
    #                  meuc[:, 0] - 2*sa[:, 0],
    #                  meuc[:, 0] + 2*sa[:, 0],
    #                  color='C1', alpha=0.2)
    #
    # # Plot the other samples
    # plt.plot(X, np.hstack([*samps]))

    # # Check whether w kernel is closer to empirical covariance matrix than the other juan
    # # empirical_cov = np.cov(gp_samples[1])
    true_gp_cov = cov_func(X).eval()
    kbar_diff_norm = np.linalg.norm(true_gp_cov - Kbar)
    keuc_diff_norm = np.linalg.norm(true_gp_cov - Keuc)
    print("KBAR diff norm", kbar_diff_norm)
    print("Keuc diff norm", keuc_diff_norm)

    return kbar_diff_norm, keuc_diff_norm

def run():

    kbd_list = []
    ked_list = []
    for seed in [1, 2, 3]:
        for outlier_mag in [0.0, 0.1, 0.5, 1.0, 10.0]:
            kbd, ked = single_exp(12, outlier_mag, seed)
            kbd_list.append((kbd, outlier_mag, seed))
            kbe_list.append((kbe, outlier_mag, seed))

    return kbd_list, kbe_list

if __name__ == '__main__':
    run()
