# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
"""Unit tests for cibin functions."""

import pytest
import numpy as np
import unittest
from cibin import *


# -

def test_nchoosem_bad_input():
    """
    Test that sample size (m) is less than or equal to
    possible values (n).
    """
    with pytest.raises(AssertionError):
        nchoosem(10, 20)


def test_nchoosem_correct_output():
    """
    Test that nchoosem produces accurate output matrix.
    """
    Z = nchoosem(3, 2)
    assert(np.all(Z[0] == [1,1,0]))
    assert(np.all(Z[1] == [1,0,1]))
    assert(np.all(Z[2] == [0,1,1]))


def test_pval_two_bad_input():
    """
    Test that number of subjects who are 1 (m) must be less
    than or equal to sum of all subjects (n).
    """
    Z_all = nchoosem(n, m)
    tau_obs = n11/m - n01/(n-m)
    with pytest.raises(AssertionError):
        pval_two(5, 10, np.array([1,2,3,4]), Z_all, tau_obs)


def test_pval_two_correct_output():
    """
    Test that pval_two produces correct p-value.
    """
    n11 = 4
    n01 = 3
    n00 = 2
    n10 = 1
    m = n10 + n11
    n = n11 + n01 + n00 + n10
    N11 = 3
    Z_all = nchoosem(n, m)
    tau_obs = n11/m - n01/(n-m)

    output = pval_two(n, m, np.array([1,2,3,4]), Z_all, tau_obs)
    assert (0.364 <= output <= 0.366)
