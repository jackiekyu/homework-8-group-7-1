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

    
def test_comb_bad_input():
    """
    Test that sample size (m) is less than or equal to
    possible values (n).
    """
    with pytest.raises(AssertionError):
        comb(5, 6, 5)
    
    
def test_comb_correct_output():
    """
    Test that function calculates a sample
    of the correct size.
    """
    n = 5
    m = 3
    nperm = 2
    Z = comb(n, m, nperm)
    assert(len(Z) == nperm)

    
def test_pval_two_bad_input():
    """
    Test that number of subjects who are 1 (m) must be less
    than or equal to sum of all subjects (n).
    """
    n = 5
    m = 10
    n11 = 4
    n01 = 3
    tau_obs = n11/m - n01/(n-m)
    with pytest.raises(AssertionError):
        pval_two(5, 10, np.array([1,2,3,4]), nchoosem(n, m), tau_obs)


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


def test_check_compatible_bad_input():
    """
    Test that n11, n01, n00, and n10 are all
    integer values.
    """
    n11 = 1.2
    n01 = 5.4
    n00 = 12.6
    n10 = 13.2
    N11 = np.array([5, 6])
    N10 = np.array([6, 8])
    N01 = np.array([7, 8])
    
    with pytest.raises(AssertionError):
        check_compatible(n11, n10, n01, n00, N11, N10, N01)


def test_check_compatible_correct_output():
    """
    Test that check_compatible() computes 
    compatibility correctly.
    """
    output = check_compatible(1, 5, 12, 13, np.array([5, 6]), np.array([6,8]), np.array([7, 8]))
    assert (np.all(output))


def test_tau_lower_N11_twoside_bad_input():
    """
    Test that n11, n01, n00, and n10 are all
    integer values.
    """
    n11 = 4.2
    n01 = 3.1
    n00 = 2
    n10 = 1
    m = 5
    n = 10
    N11 = 3
    Z_all = nchoosem(n, m)
    alpha = 0.05
    
    with pytest.raises(AssertionError):
        tau_lower_N11_twoside(n11, n10, n01, n00, N11, Z_all, alpha)


def test_tau_lower_N11_twoside_correct_output():
    """
    Test that function correctly calculates outputs,
    specifically tau min, tau max, lower accept region, 
    upper accept region, and number of total tests run.
    """
    n11 = 4
    n01 = 3
    n00 = 2
    n10 = 1
    m = n11 + n10
    n = n11 + n01 + n00 + n10
    N11 = 3
    Z_all = nchoosem(n, m)
    alpha = 0.05
    
    dict = tau_lower_N11_twoside(n11, n10, n01, n00, N11, Z_all, .05)
    assert (dict['tau_min'] == -0.3)
    assert (dict['tau_max'] == 0.2)
    assert (np.all(dict['N_accept_min'] == [3,1,4,2]))
    assert (np.all(dict['N_accept_max'] == [3,1,4,2]))
    assert (dict['rand_test_num'] == 8)


def test_tau_twoside_lower_bad_input():
    """
    Test that n11, n01, n00, and n10 are all
    integer values.
    """
    n11 = 4.2
    n01 = 3
    n00 = 2.1
    n10 = 1
    m = 5
    n = 10
    Z_all = nchoosem(n, m)
    alpha = 0.05
    
    with pytest.raises(AssertionError):
        tau_twoside_lower(n11, n10, n01, n00, Z_all, alpha)
        

def tau_twoside_less_treated_bad_input():
    """
    Test that n11, n01, n00, and n10 are all
    integer values.
    """
    n11 = 4.2
    n01 = 3
    n00 = 2
    n10 = 1.5
    m = n10 + n11
    n = n11 + n01 + n00 + n10
    N11 = 3
    Z_all = nchoosem(n, m)
    tau_obs = n11/m - n01/(n-m)
    
    with pytest.raises(AssertionError):
        tau_twoside_less_treated(n11, n10, n01, n00, 0.05, n)
    
    
def tau_twoside_less_treated_correct_output():
    """
    Test that function correctly calculates outputs,
    specifically tau min, tau max, lower accept region, 
    upper accept region, and number of total tests run.
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
    
    dict = tau_twoside_less_treated(n11, n10, n01, n00, 0.05, n)
    assert (dict['tau_lower'] == -0.4)
    assert (dict['tau_upper'] == 0.6)
    assert (np.all(dict['N_accept_lower'] == [4,0,4,2]))
    assert (np.all(dict['N_accept_upper'] == [3,6,0,1]))
    assert (dict['rand_test_total'] == 90)


def tau_twosided_ci_bad_input():
    """
    Tests that function recognizes computational
    price by raising an exception if it hits the 
    maximum number of combinations.
    """
    with pytest.raises(ValueError):
        tau_twosided_ci(1, 1, 1, 13, .05, exact=True, max_combinations=2)

def tau_twosided_ci_correct_output_exact_False():
    """
    Tests that function calculates confidence
    intervals within appropriate range when 
    parameter 'exact' is set to False.
    """
    output = tau_twosided_ci(1, 1, 1, 13, .05, exact=False, 
                             max_combinations=10**5, reps=500)
    assert (output[0] == [-1.0, 14.0])
    assert (np.all(output[1][0] == np.array([ 1.,  0.,  1., 14.])))
    assert (np.all(output[1][1] == np.array([ 1.,  14.,  0., 1.])))
    assert (output[2] == [103,500])
    
    
def tau_twosided_ci_correct_output_exact_True():
    """
    Tests that function calculates confidence
    intervals within appropriate range when 
    parameter 'exact' is set to True.
    """
    output = tau_twosided_ci(1, 1, 1, 13, .05, 
                             exact=True, max_combinations=10**5)
    assert (output[0] == [-1.0, 14.0])
    assert (np.all(output[1][0] == np.array([ 1.,  0.,  1., 14.])))
    assert (np.all(output[1][1] == np.array([ 1.,  14.,  0., 1.])))
    assert (output[2] == [103, 120])
    
    
