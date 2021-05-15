"""Translate R code for Li and Ding paper."""

import numpy as np
import pandas as pd
import math
from itertools import combinations
from itertools import filterfalse
import math


def nchoosem(n, m):
    """blurb here"""
    c = math.comb(n, m)
    trt = np.array(list(combinations(np.arange(n), m)))
    Z = np.zeros((c, n))
    for i in np.arange(c):
        Z[i, trt[i, :]] = 1
    return Z


def pval_two(n, m, N, Z_all, tau_obs):
    """blurb here"""
    n_Z_all = Z_all.shape[0]
    dat = np.zeros((n, 2))
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
    tau_hat = np.matmul(Z_all, dat[:, 0])/(m) - np.matmul((1 - Z_all), dat[:, 1])/(n-m)
    tau_N = (N[1]-N[2])/n 
    pd = sum(np.round(np.abs(tau_hat-tau_N),15)>=np.round(np.abs(tau_obs-tau_N),15))/n_Z_all
    return pd


def check_compatible(n11, n10, n01, n00, N11, N10, N01):
    """blurb here"""
    n = n11 + n10 + n01 + n00
    n_t = len(N10)
    left = np.max(np.array([np.repeat(0, n_t), n11 - np.array(N10), np.array(N11) - n01, np.array(N11) + np.array(N01)-n10-n01]), axis=0)
    right = np.min(np.array([np.array(N11), np.repeat(n11, n_t), np.array(N11) + np.array(N01) - n01, n-np.array(N10)-n01-n10]), axis=0)
    compat = left <= right
    return list(compat)


def tau_lower_N11_twoside(n11, n10, n01, n00, N11, Z_all, alpha):
    """blurb here"""
    n = n11 + n10 + n01 + n00
    m = n11 + n10
    tau_obs = n11 / m - n01 / (n - m)
    ntau_obs = n * n11 / m - n * n01 / (n - m)
    # N01 range from max((-n*tau_obs),0) to n-N11
    N10 = 0
    N01_vec0 = np.arange(0, (n-N11))[np.arange(0, (n-N11)) >= (-ntau_obs)] #  check if c() is inclusive
    N01 = min(N01_vec0)
    M = np.repeat(np.nan, len(N01_vec0))
    ### need to change
    ### counting number of randomization test
    rand_test_num = 0
    while (N10 <= (n - N11 - N01) and N01 <= (n - N11)):
        if N10 <= (N01 + ntau_obs):
            pl = pval_two(n, m, np.array([N11, N10, N01, n - (N11 + N10 + N01)]), Z_all, tau_obs)
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
        N10_upper = min((n - N11_vec0[i] - N01_vec0[i]), np.floor(N01_vec0[i] + ntau_obs))
        if N10_vec0[i] <= N10_upper:
            N10_vec = np.append(N10_vec, np.array(np.arange(N10_vec0[i], N10_upper + 1)))
            N11_vec = np.append(N11_vec, np.repeat(N11_vec0[i], (N10_upper-N10_vec0[i]+1)))
            N01_vec = np.append(N01_vec, np.repeat(N01_vec0[i], (N10_upper-N10_vec0[i]+1)))

    compat = check_compatible(n11, n10, n01, n00, N11_vec, N10_vec, N01_vec)
    
    if sum(compat) > 0:
        tau_min = min(N10_vec[compat] - N01_vec[compat]) / n
        accept_pos = np.where((N10_vec[compat] - N01_vec[compat]) == n * tau_min)
        accept_pos = accept_pos[0]
        N_accept_min = np.array([N11, N10_vec[compat][accept_pos][0], N01_vec[compat][accept_pos][0], n-(N11+N10_vec[compat][accept_pos]+N01_vec[compat][accept_pos])[0]])
        tau_max = max(N10_vec[compat] - N01_vec[compat]) / n
        accept_pos = np.where((N10_vec[compat] - N01_vec[compat]) == n * tau_min)
        accept_pos = accept_pos[0]
        N_accept_max = np.array([N11, N10_vec[compat][accept_pos][0], N01_vec[compat][accept_pos][0], n-(N11+N10_vec[compat][accept_pos]+N01_vec[compat][accept_pos])[0]])
    else:
        tau_min = math.inf
        N_accept_min = np.nan
        tau_max = -math.inf
        N_accept_max = np.nan
    return {"tau_min": tau_min, "tau_max": tau_max, "N_accept_min": N_accept_min, "N_accept_max":N_accept_max, "rand_test_num":rand_test_num}

