"""Sterne CI functions."""

import math
import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom, hypergeom

def hypergeom_conf_interval(
        n,
        x,
        N,
        cl=0.975,
        alternative="two-sided",
        G=None):
    """
    Confidence interval for a hypergeometric distribution parameter G.

    Based on the number x of good objects in a simple random sample
    of size n. The method argument specifies how to calculate the bounds.

    Parameters
    ----------
    n : int
        The number of draws without replacement.
    x : int
        The number of "good" objects in the sample.
    N : int
        The number of objects in the population.
    cl : float in (0, 1)
        The desired confidence level.
    alternative : {"two-sided", "lower", "upper"}
        Indicates the alternative hypothesis.
    G : int in [0, N]
        Starting point in search for confidence bounds for the
        hypergeometric parameter G.

    Returns
    -------
    tuple
        lower and upper confidence level with coverage (at least)
        1-alpha.
    """
    assert alternative in (
        "two-sided", "lower", "upper"), 'invalid alternative.'

    if n < x:
        raise ValueError(
            "Cannot observe more good elements than the sample size")
    if N < n:
        raise ValueError("Population size cannot be smaller than sample size")
    if n == 0:
        raise ValueError("Sample size cannot be equal to 0")
    if x < 0:
        raise ValueError("Cannot observe negative number of good elements")

    assert 0 < cl < 1, 'The confidence level must be between 0 and 1'

    if G is not None:
        if N < G:
            raise ValueError(
                "Number of good elements can't exceed the population size")
        if G < 0:
            raise ValueError(
                "G must be non-negative")
    else:
        # Initialize values for G if not provided
        # Using proportion of observed good elements x in sample size n
        # Multiply proportion by population size to get estimate for G
        G = (x / n) * N

    # Initialize values for upper ci bound (N max) and lower ci bound (0 min)
    ci_low = 0
    ci_upp = N

    # Sterne method as implemented in
    # https://ucb-stat-159-s21.github.io/site/Notes/confidence-sets.html
    # define alpha based on confidence level
    
    if alternative == 'two-sided':
        cl = 1 - (1 - cl) / 2
    
    alpha = 1 - cl

    # Lower confidence interval bound calculation
    if alternative != "upper" and x > 0:
        # Adjust lower bound until x is in the acceptance region.
        # Helper function returns accepted values based on n
        # and probability of success in each trial, ci_low.
        while x not in hypergeom_acceptance_region(n, ci_low, N, alpha):
            ci_low += 1
            if ci_low > n:
                ci_low = n
                break

    # Higher confidence interval bound calculation
    if alternative != "lower" and x < n:
        # Adjust upper bound until x is in the acceptance region.
        # Helper function returns accepted values based on n
        # and probability of success in each trial, ci_upp.
        while x not in hypergeom_acceptance_region(n, ci_upp, N, alpha):
            # Lower probability of success by small amount eps
            ci_upp -= 1
            if ci_low < 0:
                ci_low = 0
                break

    return math.ceil(ci_low), math.floor(ci_upp)


def hypergeom_acceptance_region(n, G, N, alpha=0.05):
    """
    Helper function.

    For calculating the acceptance region
    for the Sterne method of finding confidence intervals.

    Parameters
    ----------
    n : integer
        number of independent trials
    G : integer
        number of good objects in population
    N : int
        The number of objects in the population.
    alpha : float in (0,1)
        desired significance level
    Returns
    -------
    I_list : list
        values for which the test does not reject
    """
    # start with all possible outcomes (then remove some)
    pos_out = min(G, n)
    x = np.arange(0, pos_out + 1)
    I_list = list(x)  # use .tolist() instead?

    # probability mass function for each possible x value
    pmf = hypergeom.pmf(x, N, G, n)
    # smallest and largest outcome still in I
    bottom = 0
    top = pos_out
    # outcomes for which the test is randomized
    J = []
    # probability of outcomes for which test is randomized
    p_J = 0
    # probability of outcomes excluded from I
    p_tail = 0

    # need to remove outcomes from the acceptance region
    while p_tail < alpha:
        # find probability of bottom and top value in acceptance region
        pb = pmf[bottom]
        pt = pmf[top]

        # the smaller possibility has smaller probability
        if pb < pt:
            J = [bottom]
            p_J = pb
            bottom += 1

        # the larger possibility has smaller probability
        elif pb > pt:
            J = [top]
            p_J = pt
            top -= 1

        # the two possibilities have equal probability
        else:
            if bottom < top:
                J = [bottom, top]
                p_J = pb + pt
                # adjust bottom and top
                bottom += 1
                top -= 1
            else:
                # there is only one possibility left
                J = [bottom]
                p_J = pb
                bottom += 1

        # adjust p_tail
        p_tail += p_J

        # remove outcomes from acceptance region
        for j in J:
            I_list.remove(j)
    return_val = None
    return_val = I_list

    return return_val
