"""Translate R code to Python."""

import numpy as np
import scipy.special
import pandas as pd
import math
from itertools import combinations
from itertools import filterfalse
import math


def nchoosem(n, m):
    """
    Print out all possible patterns of n choose m.

    Parameters
    ----------
    n: int
       possible values to choose from
    m: int
       sample size of unordered values

    Returns
    -------
    Z: matrix
        all possible combinations of n choose m
    """
    assert m <= n, "m must be less than or equal to n."

    c = math.comb(n, m)
    trt = np.array(list(combinations(np.arange(n), m)))
    Z = np.zeros((c, n))
    for i in np.arange(c):
        Z[i, trt[i, :]] = 1
    return Z


def comb(n, m, nperm):
    """
    FIX.

    Calculate the chi squared statistic between x and y.

    Acceptance region for a randomized binomial test.

    Parameters
    ----------
    n : integer
        number of independent trials
    p : float
        probability of success in each trial
    alpha : float
        desired significance level

    Returns
    --------
    B : list
        values for which the test does not reject
    """
    assert m <= n, "m must be less than or equal to n."

    trt = np.zeros((nperm, m), dtype=int)
    for i in np.arange(0, nperm):
        trt[i, ] = np.random.choice(n, size=m, replace=False)
    Z = np.zeros((nperm, n), dtype=int)
    for i in np.arange(0, nperm):
        Z[i, trt[i, ]] = 1
        Z[i, (~np.in1d(np.arange(Z.shape[1]), trt[i, ])).nonzero()] = 0
    return Z


def pval_two(n, m, N, Z_all, tau_obs):
    """
    Calculate the p-value of a two sided test.

    Given a tau_obs value use absolute value to
    find values more extreme than the observed tau.

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
    assert m <= n, "m must be less than or equal to n"
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
    tau_N = (N[1]-N[2]) / n
    pd = sum(np.round(np.abs(tau_hat-tau_N), 15) >=
             np.round(np.abs(tau_obs-tau_N), 15))/n_Z_all
    return pd


def check_compatible(n11, n10, n01, n00, N11, N10, N01):
    """
    Helper function for tau_lower_N11_twoside.

    Checking to see if the inputs of the subject
    groups are able to be passed in correctly.

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
    N10 : array
        values of all n10
    N01 : array
        values of all n01

    Returns
    --------
    compat : list
        True or False values of compatible inputs
    """
    assert isinstance(n11, int), "n11 must be an integer."
    assert isinstance(n10, int), "n10 must be an integer."
    assert isinstance(n01, int), "n01 must be an integer."
    assert isinstance(n00, int), "n00 must be an integer."

    n = n11 + n10 + n01 + n00
    n_t = len(N10)
    left = np.max(np.array([np.repeat(0, n_t), n11 -
                            np.array(N10), np.array(N11) -
                            n01, np.array(N11) + np.array(N01)-n10-n01]),
                  axis=0)
    right = np.min(np.array([np.array(N11),
                             np.repeat(n11, n_t),
                             np.array(N11) + np.array(N01) - n01,
                             n-np.array(N10)-n01-n10]), axis=0)
    compat = left <= right
    return list(compat)


def tau_lower_N11_twoside(n11, n10, n01, n00, N11, Z_all, alpha):
    """
    Approximating tau given a set of sample size inputs.

    Calculate the lower and upper bounds for approximating
    the value of tau. Also provide the number of tests ran
    for the function and the arrays where the bounds for the
    upper and lower values were found.

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
        dictionary of values of tau min, tau max,
        lower accept region, upper accept region,
        and total tests ran
    """
    assert isinstance(n11, int), "n11 must be an integer."
    assert isinstance(n10, int), "n10 must be an integer."
    assert isinstance(n01, int), "n01 must be an integer."
    assert isinstance(n00, int), "n00 must be an integer."

    n = n11 + n10 + n01 + n00
    m = n11 + n10
    tau_obs = n11 / m - n01 / (n - m)
    ntau_obs = n * n11 / m - n * n01 / (n - m)
    # N01 range from max((-n*tau_obs),0) to n-N11
    N10 = 0
    N01_vec0 = np.arange(0, (n-N11)+1)[np.arange(0, (n-N11)+1) >= (-ntau_obs)]
    N01 = min(N01_vec0)
    M = np.repeat(np.nan, len(N01_vec0))
    # counting number of randomization test
    rand_test_num = 0
    while (N10 <= (n - N11 - N01) and N01 <= (n - N11)):
        if N10 <= (N01 + ntau_obs):
            pl = pval_two(n, m, np.array([N11,
                                          N10, N01,
                                          n - (N11 + N10 + N01)]),
                          Z_all, tau_obs)
            rand_test_num += 1
            if pl >= alpha:
                M[N01_vec0 == N01] = N10
                N01 = N01 + 1
            else:
                N10 = N10 + 1
        else:
            M[N01_vec0 == N01] = N10
            N01 = N01 + 1
    if N01 <= (n - N11):
        M[N01_vec0 >= N01] = np.floor(N01_vec0[N01_vec0 >= N01] + ntau_obs) + 1
    N11_vec0 = np.repeat(N11, len(N01_vec0))
    N10_vec0 = M
    N11_vec = np.array([])
    N10_vec = np.array([])
    N01_vec = np.array([])
    for i in np.arange(len(N11_vec0)):
        N10_upper = min((n - N11_vec0[i] - N01_vec0[i]),
                        np.floor(N01_vec0[i] + ntau_obs))
        if N10_vec0[i] <= N10_upper:
            N10_vec = np.append(N10_vec,
                                np.array(np.arange(N10_vec0[i],
                                                   N10_upper + 1)))
            N11_vec = np.append(N11_vec,
                                np.repeat(N11_vec0[i],
                                          (N10_upper-N10_vec0[i]+1)))
            N01_vec = np.append(N01_vec,
                                np.repeat(N01_vec0[i],
                                          (N10_upper-N10_vec0[i]+1)))

    compat = check_compatible(n11, n10, n01, n00, N11_vec, N10_vec, N01_vec)

    if sum(compat) > 0:
        tau_min = min(N10_vec[compat] - N01_vec[compat]) / n
        accept_pos = np.where((N10_vec[compat] - N01_vec[compat])
                              == n * tau_min)
        accept_pos = accept_pos[0]
        N_accept_min = np.array([N11, N10_vec[compat][accept_pos][0],
                                 N01_vec[compat][accept_pos][0],
                                 n-(N11+N10_vec[compat][accept_pos] +
                                    N01_vec[compat][accept_pos])[0]])
        tau_max = max(N10_vec[compat] - N01_vec[compat]) / n
        accept_pos = np.where((N10_vec[compat] - N01_vec[compat])
                              == n * tau_min)
        accept_pos = accept_pos[0]
        N_accept_max = np.array([N11, N10_vec[compat][accept_pos][0],
                                 N01_vec[compat][accept_pos][0],
                                 n-(N11+N10_vec[compat][accept_pos] +
                                    N01_vec[compat][accept_pos])[0]])
    else:
        tau_min = math.inf
        N_accept_min = np.nan
        tau_max = -math.inf
        N_accept_max = np.nan
    return {"tau_min": tau_min,
            "tau_max": tau_max,
            "N_accept_min": N_accept_min,
            "N_accept_max": N_accept_max,
            "rand_test_num": rand_test_num}


def tau_twoside_lower(n11, n10, n01, n00, alpha, Z_all):
    """
    FIX..

    Checking to see if the inputs of the subject
    groups are able to be passed in correctly.

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
    compat : list
        True or False values of compatible inputs
    """
    assert isinstance(n11, int), "n11 must be an integer."
    assert isinstance(n10, int), "n10 must be an integer."
    assert isinstance(n01, int), "n01 must be an integer."
    assert isinstance(n00, int), "n00 must be an integer."

    n = n11+n10+n01+n00
    m = n11+n10
    tau_obs = n11/m - n01/(n-m)
    ntau_obs = n * n11 / m - n * n01 / (n - m)
    tau_min = math.inf
    tau_max = -math.inf
    N_accept_min = np.nan
    N_accept_max = np.nan
    rand_test_total = 0

    for N11 in np.arange(0, min((n11+n01), n+ntau_obs)+1):
        N01_vec0 = np.arange(0, n-N11+1)[np.arange(0, (n-N11)+1) >= (-ntau_obs)]
        if len(list(N01_vec0)) == 0:
            break
        tau_min_N11 = tau_lower_N11_twoside(n11, n10, n01, n00, N11,
                                            Z_all, alpha)
        # assumes that tau_lower_N11_twoside output is a dictionary
        rand_test_total = rand_test_total + tau_min_N11["rand_test_num"]
        if(tau_min_N11["tau_min"] < tau_min):
            N_accept_min = tau_min_N11["N_accept_min"]
        if(tau_min_N11["tau_max"] > tau_max):
            N_accept_max = tau_min_N11["N_accept_max"]
        tau_min = min(tau_min, tau_min_N11["tau_min"])
        tau_max = max(tau_max, tau_min_N11["tau_max"])

    tau_lower = tau_min
    tau_upper = tau_max
    N_accept_lower = N_accept_min
    N_accept_upper = N_accept_max

    dict_output = {'tau_lower': tau_lower, 'N_accept_lower': N_accept_lower,
                   'tau_upper': tau_upper, 'N_accept_upper': N_accept_upper,
                   'rand_test_total': rand_test_total}
    return dict_output


def tau_twoside_less_treated(n11, n10, n01, n00, alpha, nperm):
    """
    FIX..

    Checking to see if the inputs of the subject
    groups are able to be passed in correctly.

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
    compat : list
        True or False values of compatible inputs
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

    ci_lower = tau_twoside_lower(n11, n10, n01, n00, alpha, Z_all)
    ci_upper = tau_twoside_lower(n10, n11, n00, n01, alpha, Z_all)

    # this implementation depends on tau_twoside_lower returning a dictionary
    rand_test_total = ci_lower["rand_test_total"] + ci_upper["rand_test_total"]

    tau_lower = min(ci_lower["tau_lower"], -1 * ci_upper["tau_upper"])
    tau_upper = max(ci_lower["tau_upper"], -1 * ci_upper["tau_lower"])

    if tau_lower == ci_lower["tau_lower"]:
        N_accept_lower = ci_lower["N_accept_lower"]
    else:
        # reverse N_accept_upper
        N_accept_lower = np.flipud(ci_upper["N_accept_upper"])

    if tau_upper == -1 * ci_upper["tau_lower"]:
        # reverse N_accept_lower
        N_accept_upper = np.flipud(ci_upper["N_accept_lower"])
    else:
        N_accept_upper = ci_lower["N_accept_upper"]

    return {"tau_lower": tau_lower,
            "tau_upper": tau_upper,
            "N_accept_lower": N_accept_lower,
            "N_accept_upper": N_accept_upper,
            "rand_test_total": rand_test_total}


def tau_twosided_ci(n11, n10, n01, n00, alpha,
                    exact=True, max_combinations=10**5, reps=10**3):
    """
    FIX.

    Checking to see if the inputs of the subject
    groups are able to be passed in correctly.

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
    compat : list
        True or False values of compatible inputs
    """
    n = n11 + n10 + n01 + n00
    m = n11 + n10

    if exact is True:
        reps = scipy.special.comb(n, m, exact=True)
        if reps > max_combinations:
            raise ValueError(
                "Number of reps can't exceed max_combinations")

    if m > (n/2):
        ci = tau_twoside_less_treated(n01, n00, n11, n10, alpha, reps)
        tau_lower = -ci["tau_upper"]
        tau_upper = -ci["tau_lower"]
        N_accept_lower = ci["N_accept_lower"][[0, 2, 1, 3]]
        N_accept_upper = ci["N_accept_upper"][[0, 2, 1, 3]]
        rand_test_total = ci["rand_test_total"]
    else:
        ci = tau_twoside_less_treated(n11, n10, n01, n00, alpha, reps)
        tau_lower = ci["tau_lower"]
        tau_upper = ci["tau_upper"]
        N_accept_lower = ci["N_accept_lower"]
        N_accept_upper = ci["N_accept_upper"]
        rand_test_total = ci["rand_test_total"]

    bounds = [tau_lower*n, tau_upper*n]
    allocation = [N_accept_lower, N_accept_upper]
    tables_reps = [rand_test_total, reps]
    return bounds, allocation, tables_reps
