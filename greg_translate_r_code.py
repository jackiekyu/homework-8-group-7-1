"""Translate R code for Li and Ding paper."""

import numpy as np
import pandas as pd


def pval_two(n, m, N, Z_all, tau_obs):
    n_Z_all = Z_all[0]
    dat = np.zeros((n, 2))
    if N[0] > 0:
        dat[0:N[0], ] = 1
    if N[1] > 0:
        dat[(N[0] + 1): (N[0] + N[1]), 0] = 1
        dat[(N[0] + 1): (N[0] + N[1]), 1] = 0
    if N[2] > 0:
        dat[(N[0]+N[1]+1):(N[0]+N[1]+N[2]), 0] = 0
        dat[(N[0]+N[1]+1):(N[0]+N[1]+N[2]), 1] = 1
    if N[3] > 0:
        dat[(N[0]+N[1]+N[2]+1):(N[0]+N[1]+N[2]+N[3]), ] = 0
    tau_hat = np.matrix(Z_all) * np.matrix(dat[:, 0]/(m)) - (1 - np.matrix(Z_all)) * np.matrix(dat[:, 1]/(n - m))
    pd = sum(round(abs(tau_hat - tau_N), 15) >= round(abs(tau_obs - tau_N), 15)) / n_Z_all
    return pd

