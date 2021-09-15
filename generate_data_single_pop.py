#!/usr/bin/env python
"""
Generates data from a single population
"""

import sys, os
import argparse
import pickle
import logging

import scipy.stats
import numpy as np
import pandas as pd

from dataset import *

def parse_args():
    parser = argparse.ArgumentParser(description='run simulation')
    parser.add_argument(
        '--meta-seed',
        type=int,
        default=0,
        help='seed for initial meta data')
    parser.add_argument(
        '--data-seed',
        type=int,
        default=0,
        help='seed for data stream')
    parser.add_argument(
        '--change-p',
        type=int,
        default=4,
        help="number of coefficients to perturb when a shift occurs")
    parser.add_argument(
        '--sparse-p',
        type=int,
        default=4,
        help="number of coefficients with initial values that are large")
    parser.add_argument(
        '--p',
        type=int,
        default=10,
        help="total number of coefficients")
    parser.add_argument(
        '--init-sparse-beta',
        type=float,
        default=0.5,
        help="(negative) mean value of the true initial coefficients")
    parser.add_argument(
        '--init-recalib-n',
        type=int,
        default=300,
        help="number of datapoints for initial recalibration of initial model")
    parser.add_argument(
        '--init-train-n',
        type=int,
        default=1000,
        help="number of datapoints for training initial model")
    parser.add_argument(
        '--init-perturb',
        type=float,
        default=0,
        help="how much to perturb the coefficients at time zero")
    parser.add_argument(
        '--test-n',
        type=int,
        default=2000,
        help="number of test observations")
    parser.add_argument(
        '--simulation',
        type=str,
        default="tiny",
        help="string specifying simulation setting, simulation options in constants.py")
    parser.add_argument(
        '--out-file',
        type=str,
        default="_output/data.pkl",
        help="output file for data")
    parser.add_argument(
        '--log-file',
        type=str,
        default="_output/log.txt",
        help="log file")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    # parameters
    np.random.seed(args.meta_seed)

    # Prep data
    init_beta = np.random.normal(size=(args.p,1)) * 0.05
    init_beta[:args.sparse_p] += -args.init_sparse_beta
    data_generator = DataGenerator(init_beta, 0, init_perturb=args.init_perturb, meta_seed=args.meta_seed, data_seed=args.data_seed, simulation=args.simulation)
    full_dat, beta_time_varying = data_generator.generate_data(args.init_train_n, args.init_recalib_n, args.test_n, args.change_p)
    print("class", np.mean(full_dat.merged_dat.y))
    print("MU DIFF", np.mean(np.abs(full_dat.merged_dat.mu - 0.5)))
    full_multi_pop_data = FullDatasetMultiPop([full_dat], np.array([1]), meta_seed=args.meta_seed)
    logging.info("num batches %d", full_multi_pop_data.num_batches)

    with open(args.out_file, "wb") as f:
        pickle.dump({
            "full_dat": full_multi_pop_data,
            "betas": [beta_time_varying],
            }, f)


if __name__ == "__main__":
    main()
