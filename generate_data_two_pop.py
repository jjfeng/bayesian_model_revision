#!/usr/bin/env python
"""
Generate data for a population with two subpops
"""

import sys, os
import argparse
import pickle
import logging

import scipy.stats
import numpy as np
import pandas as pd

from dataset import *
from constants import SIM_SETTINGS

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
        '--subpopulations',
        type=str,
        default="20,80",
        help='comma-separated percentages for each subpopulation')
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.6,
        help="mean value of initial coefficients for one set of variables")
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.5,
        help="mean value of initial coefficients for a second set of variables")
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
        '--init-perturbs',
        type=str,
        default="0,0",
        help="comma-separated for the two subpops, how much to perturb the coefficients at time zero")
    parser.add_argument(
        '--test-n',
        type=int,
        default=2000,
        help="number of test observations")
    parser.add_argument(
        '--simulation',
        type=str,
        default="new_iid,new_iid",
        help="comma-separated string specifying simulation setting for the two subpopulations, simulation options in constants.py")
    parser.add_argument(
        '--out-file',
        type=str,
        default="_output/data.pkl",
        help="output file for data",
    parser.add_argument(
        '--log-file',
        type=str,
        default="_output/log.txt",
        help="log file")
    args = parser.parse_args()
    args.init_perturbs= np.array(list(map(float, args.init_perturbs.split(","))))
    args.subpopulations = np.array(list(map(float, args.subpopulations.split(","))))
    args.subpopulations /= np.sum(args.subpopulations)
    args.simulation = args.simulation.split(",")
    print(args.subpopulations)
    # This is to ensure that both populations will get the same set of coefficients, though shuffled in order
    assert args.sparse_p % 2 == 0
    return args

def main():
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)

    # Prep data
    all_dats = []
    all_betas = []
    for pop_idx, (subpop, sim_pop) in enumerate(zip(args.subpopulations, args.simulation)):
        print("SIMULATION", sim_pop)
        mean_x = 0
        init_perturb = args.init_perturbs[pop_idx]

        # Vary the coefficients in the two populations
        init_beta = np.zeros((args.p,1))
        # The two populations have similar coefficients, but not exactly the same
        init_beta[:int(args.sparse_p/2)] = -(args.beta1 * (1 - pop_idx) + args.beta2 * pop_idx)
        init_beta[int(args.sparse_p/2):args.sparse_p] = -(args.beta2 * (1 - pop_idx) + args.beta1 * pop_idx)
        data_generator = DataGenerator(init_beta, mean_x, init_perturb=init_perturb, meta_seed=args.meta_seed, data_seed=args.data_seed, simulation=sim_pop, subpop=pop_idx, num_pops=2)
        full_dat, beta_time_varying = data_generator.generate_data(args.init_train_n, args.init_recalib_n, args.test_n, change_p=args.change_p)
        all_betas.append(beta_time_varying)
        all_dats.append(full_dat)
    full_multi_pop_data = FullDatasetMultiPop(all_dats, args.subpopulations, meta_seed=args.meta_seed)

    with open(args.out_file, "wb") as f:
        pickle.dump({
            "full_dat": full_multi_pop_data,
            "betas": all_betas,
            }, f)


if __name__ == "__main__":
    main()
