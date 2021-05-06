"""Translate R code to Python."""

import numpy as np

def comb(n, m, nperm):
    trt = np.zeros((nperm,m), dtype=int)
    for i in np.arange(0, nperm):
        trt[i,] = np.random.choice(n, size=m, replace=False)

    Z = np.zeros((nperm,n))
    for i in np.arange(0, nperm):
        Z[i,trt[i,]] = 1
        Z[i,-trt[i,]] = 0

    return Z
