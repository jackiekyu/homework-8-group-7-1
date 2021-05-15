"""Onesided CI functions."""

import numpy as np
import scipy.special
import pandas as pd
import math
from itertools import combinations
from itertools import filterfalse
import math
from cibin import *


def pval_one_lower(n, m, N, Z_all, tau_obs):
    """
    Calculate the p-value of a one sided test.

    Given a tau_obs value find values more extreme
    than the observed tau.

    Parameters
    ----------
    n : int
        the sum of all subjects in the sample group
    m : int
        number of subjects who are 1 if control group
    N : array
        an array of all subjects in all groups
    Z_all: matrix
        the output from the function nchoosem
    tau_obs: float
        the observed value of tau
    Returns
    --------
    pd : float
        the pval of the test statistic
    """
    assert m <= n, "# of subjects who are 1 must be <= to sum of all subjects"
    n_Z_all = Z_all.shape[0]
    dat = np.zeros((n, 2))
    N = [int(x) for x in N]
    if N[0] > 0:
        dat[0:N[0], :] = 1
    if N[1] > 0:
        dat[(N[0]): (N[0] + N[1]), 0] = 1
        dat[(N[0]): (N[0] + N[1]), 1] = 0
    if N[2] > 0:
        dat[(N[0]+N[1]):(N[0]+N[1]+N[2]), 0] = 0
        dat[(N[0]+N[1]):(N[0]+N[1]+N[2]), 1] = 1
    if N[3] > 0:
        dat[(N[0]+N[1]+N[2]):(N[0]+N[1]+N[2]+N[3]), ] = 0
    tau_hat = np.matmul(Z_all, dat[:, 0]) / (m) - \
        np.matmul((1 - Z_all), dat[:, 1]) / (n-m)

    pl = sum(np.round(tau_hat, 15) >= np.round(tau_obs, 15)) / n_Z_all

    return pl


def tau_lower_N11_oneside(n11, n10, n01, n00, N11, Z_all, alpha):
    """
    Approximating tau given a set of sample size inputs.

    Calculate the lower bound for approximating
    the value of tau. Also provide the array where
    the bounds for the lower value was found.

    Parameters
    ----------
    n11 : int
        number of people who fall under group n11
    n10 : int
        number of people who fall under group n10
    n01 : int
        number of people who fall under group n01
    n00 : int
        number of people who fall under group n00
    N11 : array
        values of all n11
    Z_all : matrix
        the output from the function nchoosem
    alpha : float
        the alpha cutoff value desired
    Returns
    --------
    dictionary : dictionary
        dictionary of values of tau min and
        lower accept region
    """
    assert isinstance(n11, int), "n11 must be an integer."
    assert isinstance(n10, int), "n10 must be an integer."
    assert isinstance(n01, int), "n01 must be an integer."
    assert isinstance(n00, int), "n00 must be an integer."

    n = n11 + n10 + n01 + n00
    m = n11 + n10
    N01 = 0
    N10 = 0
    tau_obs = n11 / m - n01 / (n - m)
    # N01 range from 0 to n-N11
    M = np.repeat(0, (n-N11+1))
    while (N10 <= (n - N11 - N01) and N01 <= (n - N11)):
        pl = pval_one_lower(n, m, 
                            np.array([N11, N10, N01,
                                      n - (N11 + N10 + N01)]),
                            Z_all, tau_obs)

        if pl >= alpha:
            M[N01] = N10
            N01 = N01 + 1
        else:
            N10 = N10 + 1
    if N01 <= (n - N11):
        M[(N01):(n-N11+1)] = n+1
    N11_vec0 = np.repeat(N11, (n-N11+1))
    N10_vec0 = M
    N01_vec0 = np.arange(0, (n-N11)+1)

    N11_vec = np.array([])
    N10_vec = np.array([])
    N01_vec = np.array([])
    for i in np.arange(len(N11_vec0)):
        if N10_vec0[i] <= (n-N11_vec0[i]-N01_vec0[i]):
            N10_vec = np.append(N10_vec,
                                np.array(np.arange(N10_vec0[i],
                                                   (n-N11_vec0[i]-N01_vec0[i]) + 1)))
            N11_vec = np.append(N11_vec,
                                np.repeat(N11_vec0[i],
                                          (n-N11_vec0[i]-N01_vec0[i]-N10_vec0[i]+1)))
            N01_vec = np.append(N01_vec,
                                np.repeat(N01_vec0[i],
                                          (n-N11_vec0[i]-N01_vec0[i]-N10_vec0[i]+1)))

    compat = check_compatible(n11, n10, n01, n00, N11_vec, N10_vec, N01_vec)

    if sum(compat) > 0:
        tau_min = min(N10_vec[compat] - N01_vec[compat]) / n
        accept_pos = np.where((N10_vec[compat] - N01_vec[compat])
                              == np.round(n * tau_min, 0))
        accept_pos = accept_pos[0]
        N_accept = np.array([N11, N10_vec[compat][accept_pos][0],
                                 N01_vec[compat][accept_pos][0],
                                 n-(N11+N10_vec[compat][accept_pos] +
                                    N01_vec[compat][accept_pos])[0]])
    else:
        tau_min = (n11+n00) / n 
        N_accept = np.nan
    return {"tau_min": tau_min,
            "N_accept": N_accept}


def tau_lower_oneside(n11, n10, n01, n00, alpha, nperm):
    """
    Approximating tau given a set of sample size inputs.

    Calculate the lower and upper bounds for approximating
    the value of tau for onesided test. Also provide the array
    where the bounds for the upper and lower values were found.
    Will use simulation if not possible to calculate exact.

    Parameters
    ----------
    n11 : int
        number of people who fall under group n11
    n10 : int
        number of people who fall under group n10
    n01 : int
        number of people who fall under group n01
    n00 : int
        number of people who fall under group n00
    alpha : float
        the alpha cutoff value desired
    nperm : int
       maximum number of combinations considered
    Returns
    --------
    dictionary : dict
        dictionary of values of tau lower, tau upper,
        and accept region array
    """
    assert isinstance(n11, int), "n11 must be an integer."
    assert isinstance(n10, int), "n10 must be an integer."
    assert isinstance(n01, int), "n01 must be an integer."
    assert isinstance(n00, int), "n00 must be an integer."

    n = n11 + n10 + n01 + n00
    m = n11 + n10
    if scipy.special.comb(n, m, exact=True) <= nperm:
        Z_all = nchoosem(n, m)
    else:
        Z_all = comb(n, m, nperm)

    tau_min = (n11+n00) / n
    N_accept = np.array([])

    for N11 in np.arange(0, (n11+n01)+1):
        tau_min_N11 = tau_lower_N11_oneside(n11, n10, n01, n00, N11, Z_all, alpha)
        if tau_min_N11["tau_min"] < tau_min:
            N_accept = tau_min_N11["N_accept"]
        tau_min = min(tau_min, tau_min_N11["tau_min"])

    tau_lower = tau_min
    tau_upper = (n11+n00) / n

    return {"tau_lower": tau_lower*n,
            "tau_upper": tau_upper*n,
            "N_accept": N_accept}
