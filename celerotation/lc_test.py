"""
Test Dan's celerite example on a real light curve.
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt

import celerite
from celerite import terms
from scipy.optimize import minimize
import emcee
import corner

from kepler_data import load_kepler_data


class CustomTerm(terms.Term):
    """
    The "Quasi-periodic" kernel.
    """
    parameter_names = ("log_a", "log_b", "log_tau", "log_P")

    def get_real_coefficients(self):
        b = np.exp(self.log_b)
        return (
            np.exp(self.log_a) * (1.0 + b) / (2.0 + b),
            np.exp(-self.log_tau),
        )

    def get_complex_coefficients(self):
        b = np.exp(self.log_b)
        return (
            np.exp(self.log_a) / (2.0 + b),
            0.0,
            np.exp(-self.log_tau),
            2*np.pi*np.exp(-self.log_P),
        )


def simple_kernel():
    # A non-periodic component
    Q = 1.0 / np.sqrt(2.0)
    w0 = 3.0
    S0 = np.var(y) / (w0 * Q)
    kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q),
                           log_omega0=np.log(w0),
                           bounds=[(-15, 15), (-15, 15), (-15, 15)])
    kernel.freeze_parameter("log_Q")  # We don't want to fit for "Q" in this term

    # A periodic component
    Q = 1.0
    w0 = 3.0
    S0 = np.var(y) / (w0 * Q)
    kernel += terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q),
                            log_omega0=np.log(w0),
                            bounds=[(-15, 15), (-15, 15), (-15, 15)])
    return kernel


def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)


def lnprior(params):
    log_a, log_b, log_tau, log_P = params
    if -10 < log_a < 10 and -10 < log_b < 10 and -10 < log_tau < 10 \
            and -10 < log_P < 10:
        return 0.
    else: return -np.inf


def lnprob(params, y, gp):
    return neg_log_like(params, y, gp) + lnprior(params)


def make_plots(gp, x, y, yerr):
    xs = np.linspace(min(x), max(x), 5000)
    pred_mean, pred_var = gp.predict(y, xs, return_var=True)
    pred_std = np.sqrt(pred_var)

    color = "#ff7f0e"
    plt.clf()
    plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
    plt.plot(xs, pred_mean, color=color)
    plt.fill_between(xs, pred_mean+pred_std, pred_mean-pred_std, color=color,
                     alpha=0.3, edgecolor="none")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(min(x), min(x) + 50)
    plt.savefig("{}_fit".format(id))

    omega = np.exp(np.linspace(-1, 3, 5000))
    psd = gp.kernel.get_psd(omega)

    plt.clf()
    plt.plot(omega, psd, color=color)
    for k in gp.kernel.terms:
        plt.plot(omega, k.get_psd(omega), "--", color=color)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(omega[0], omega[-1])
    plt.xlabel("$\omega$")
    plt.ylabel("$S(\omega)$")
    plt.savefig("{}_psd".format(id))

    plt.clf()
    dt = np.linspace(0, 20, 1000)
    acf = gp.kernel.get_value(dt)
    plt.plot(dt, acf, color=color)
    plt.xlabel("$\Delta$t")
    plt.ylabel("$ACF$")
    plt.savefig("{}_acf".format(id))


if __name__ == "__main__":

    LC_DIR = "/Users/ruthangus/.kplr/data/lightcurves/{}"\
        .format(str(id).zfill(9))

    # Load the data
    id = 6269070
    p_init = 20/6.

    x, y, yerr = load_kepler_data(LC_DIR)
    c = 20000  # Cut off after 1000 data points.
    x, y, yerr = x[:c], y[:c], yerr[:c]
    inds = np.argsort(x)
    x, y, yerr = x[inds], y[inds], yerr[inds]

    # Define the kernel
    kernel = CustomTerm(log_a=np.log(np.var(y)), log_b=-5, log_tau=5,
                        log_P=np.log(p_init))
    print(kernel.get_parameter_vector())

    # Generate the covariance matrix.
    gp = celerite.GP(kernel, mean=np.mean(y))
    gp.compute(x, yerr)  # You always need to call compute once.

    # # Plot a draw from the prior.
    # ys = gp.sample(x)
    # plt.clf()
    # plt.plot(x, ys)
    # plt.savefig("prior_sample")

    # Set the initial parameters and bounds.
    initial_params = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()

    # print(gp.get_parameter_names())

    # Optimise the parameters.
    start = time.time()
    r = minimize(neg_log_like, initial_params, method="L-BFGS-B",
                 bounds=bounds, args=(y, gp))
    end = time.time()
    print("time taken = ", end - start, "seconds")
    print(r.x)
    print("Period = ", np.exp(r.x[-1]), "days")
    gp.set_parameter_vector(r.x)
    make_plots(gp, x, y, yerr)

    # # Run emcee
    # ndim, nsteps = len(initial_params), 10000
    # nwalkers = 3*ndim
    # p0 = [1e-4*np.random.rand(ndim) + initial_params for i in range(nwalkers)]
    # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[y, gp])
    # pos, _, _ = sampler.run_mcmc(p0, 2000)
    # sampler.reset()
    # sampler.run_mcmc(pos, nsteps)
    # flat = np.reshape(sampler.chain, (nwalkers*nsteps, ndim))
    # fig = corner.corner(np.exp(flat), labels=["log_a", "log_b", "log_tau",
    #                                           "log_P"])
    # fig.savefig("corner_celerite")
    # results = [np.median(np.exp(flat[:, 0])), np.median(np.exp(flat[:, 1])),
    #            np.median(np.exp(flat[:, 2])), np.median(np.exp(flat[:, 3]))]
    # print(results)
