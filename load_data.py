#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Loading COPD data
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
    parser = argparse.ArgumentParser(description='load data into exepcted format')
    parser.add_argument(
        '--data-file',
        type=str,
        help='name of data file, csv')
    parser.add_argument(
        '--num-p',
        type=int,
        default=-1)
    parser.add_argument(
        '--init-recalib-n',
        type=int,
        default=300)
    parser.add_argument(
        '--init-train-n',
        type=int,
        default=1000)
    parser.add_argument(
        '--train-start-obs',
        type=int,
        default=0)
    parser.add_argument(
        '--obs-n',
        type=int,
        default=10000)
    parser.add_argument(
        '--sample-rate',
        type=float,
        default=1)
    parser.add_argument(
        '--out-file',
        type=str,
        default="_output/data.pkl")
    args = parser.parse_args()
    assert args.sample_rate <= 1
    return args

def convert_to_dataset(dat, num_p=-1, split=False):
    if split:
        all_datasets = [convert_to_dataset(dat[i: i + 1,:], num_p, split=False) for i in range(dat.shape[0])]
        return all_datasets
    else:
        x = dat[:,:num_p]
        y = dat[:,-1:].astype(int)
        return Dataset(x, y, None, None)

def main():
    args = parse_args()

    raw_dat = pd.read_csv(args.data_file, sep=",")
    timestamps = pd.to_datetime(raw_dat.admission_dt, format="%Y-%m-%d")
    dat = raw_dat.values[:,:-1].astype(float)
    print(raw_dat.copd_any.sum())
    print(dat[:,-1].sum())

    # Load data for init training and recalibration
    print("NUM train EVENTS", np.sum(dat[args.train_start_obs:args.train_start_obs + args.init_train_n,-1]))
    print("NUM recalib EVENTS", np.sum(dat[args.train_start_obs + args.init_train_n:args.train_start_obs + args.init_train_n + args.init_recalib_n,-1]))
    init_train = convert_to_dataset(dat[args.train_start_obs:args.train_start_obs + args.init_train_n,:], num_p=args.num_p)
    init_recalib = convert_to_dataset(dat[args.train_start_obs + args.init_train_n:args.train_start_obs + args.init_train_n + args.init_recalib_n,:], num_p=args.num_p)
    start_idx = args.train_start_obs + args.init_train_n + args.init_recalib_n

    # Pick out data for the monitoring data
    obs_idxs = np.arange(start_idx, start_idx + args.obs_n)
    obs_idxs = np.sort(np.random.choice(obs_idxs, size=int(obs_idxs.size * args.sample_rate), replace=False))
    print("NUM EVENTS", np.sum(dat[obs_idxs,-1]))
    streaming_dat = convert_to_dataset(dat[obs_idxs,:], num_p=args.num_p, split=True)

    # Shove all the data in the test data, so that the oracle model can use it
    test_data = convert_to_dataset(dat[obs_idxs,:], num_p=args.num_p, split=False)
    test_data_list = [test_data] + [None] * (obs_idxs.size - 1)

    full_data = FullDataset(init_train, init_recalib, streaming_dat, test_data_list=test_data_list, timestamps=timestamps[obs_idxs].to_numpy())
    full_multi_pop_data = FullDatasetMultiPop([full_data], proportions=np.array([1]), meta_seed=0)

    with open(args.out_file, "wb") as f:
        pickle.dump({
            "full_dat": full_multi_pop_data,
            }, f)


if __name__ == "__main__":
    main()

