#!/usr/bin/env python
"""
Code for calculating Type I regret bounds
"""

import sys, os
import argparse
import pickle
import logging


import scipy.stats
import numpy as np
import pandas as pd

def get_regret_bound_type_i(prior_mu, prior_sigma, prior_delta2, alpha, T: int, n:int, embedding_max_eigen):
    """
    THIS IS A TYPE I ERROR BOUND for things with a lot of shifts but a small prior_delta2

    @param: n = number of observations in a batch

    @return the best regret bound given a dynamic oracle, searches through all proxy oracles
    """
    BLR_regret = _get_regret_bound_BLR_type_i(prior_mu, prior_sigma, T, n, embedding_max_eigen)
    additional_marBLR_reg = 0
    if alpha > 0 and prior_delta2 > 0:
        num_coef = prior_mu.size
        trace = np.trace(prior_sigma)
        additional_marBLR_log_term = np.log(1 + prior_delta2 * n * embedding_max_eigen * T/2. * trace/num_coef)
        additional_marBLR_reg = num_coef/2 * alpha * (T - 1) * additional_marBLR_log_term
    return BLR_regret + additional_marBLR_reg

def _get_regret_bound_BLR_type_i(prior_mu, prior_sigma, T: int, n: int, embedding_max_eigen: float):
    """
    @return regret bound for cumulative neg log lik using a proxy oracle
    """
    num_coef = prior_mu.size
    trace = np.trace(prior_sigma)
    return num_coef/2 * np.log(1 + n * embedding_max_eigen * T * trace /num_coef)

