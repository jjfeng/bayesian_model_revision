#!/usr/bin/env python
"""
Create underlying models
"""

import sys, os
import argparse
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from dataset import Dataset
from modelers import *

def parse_args():
    parser = argparse.ArgumentParser(description='run simulation')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='seed')
    parser.add_argument(
        '--simulation',
        type=str,
        default="fixed",
        choices=["fixed", "cumulative_refit", "combo_refit", "combo_boxed", "boxed"])
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=200)
    parser.add_argument(
        '--max-depth',
        type=int,
        default=2)
    parser.add_argument(
        '--refit-freq',
        type=int,
        default=1)
    parser.add_argument(
        '--max-box',
        type=str,
        default="100")
    parser.add_argument(
        '--switch-time',
        type=int,
        default=None)
    parser.add_argument(
        '--data-file',
        type=str,
        default="_output/data.pkl")
    parser.add_argument(
        '--out-file',
        type=str,
        default="_output/models.pkl")
    args = parser.parse_args()
    args.max_box = list(map(int, args.max_box.split(",")))
    return args

def main():
    args = parse_args()
    # parameters
    np.random.seed(args.seed)

    with open(args.data_file, "rb") as f:
        data = pickle.load(f)["full_dat"]

    # Fit model
    if args.simulation == "fixed":
        clf = LockedModeler(data.get_train_dat(), max_depth=args.max_depth, n_estimators=args.n_estimators)
    elif args.simulation == "cumulative_refit":
        clf = CumulativeModeler(data.get_train_dat(), max_depth=args.max_depth, refit_freq=args.refit_freq, n_estimators=args.n_estimators)
    elif args.simulation == "boxed":
        train_dat = data.get_train_dat()
        clf = BoxedModeler(train_dat, max_depth=args.max_depth, refit_freq=args.refit_freq, max_trains=args.max_box, n_estimators=args.n_estimators, switch_time=args.switch_time)
    elif args.simulation == "combo_refit":
        train_dat = data.get_train_dat()
        clf1 = LockedModeler(train_dat, max_depth=args.max_depth, n_estimators=args.n_estimators)
        clf2 = CumulativeModeler(train_dat, max_depth=args.max_depth, refit_freq=args.refit_freq, n_estimators=args.n_estimators)
        clf = ComboModeler([clf1, clf2], args.refit_freq)
    elif args.simulation == "combo_boxed":
        train_dat = data.get_train_dat()
        clf1 = LockedModeler(train_dat, max_depth=args.max_depth, n_estimators=args.n_estimators)
        clf2 = BoxedModeler(train_dat, max_depth=args.max_depth, refit_freq=args.refit_freq, max_trains=args.max_box, n_estimators=args.n_estimators, switch_time=args.switch_time)
        clf = ComboModeler([clf1, clf2], args.refit_freq)

    with open(args.out_file, "wb") as f:
        pickle.dump(clf, f)


if __name__ == "__main__":
    main()

