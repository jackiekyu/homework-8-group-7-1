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


def tau_twoside_less_treated(n11, n10, n01, n00, alpha, nperm):
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

    return {"tau_lower": tau_lower, "tau_upper": tau_upper,  "N_accept_lower": N_accept_lower, "N_accept_upper": N_accept_upper, "rand_test_total": rand_test_total}
