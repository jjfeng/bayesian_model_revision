#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import pickle
import logging

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Summarize performance of recalibs for different hyperparam values')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='seed')
    parser.add_argument(
        '--scores-file',
        type=str,
        default="_output/calib_scores.csv")
    parser.add_argument(
        '--recalibrators-file',
        type=str,
        default="_output/recalibrators.pkl",
        help='history recalib file')
    parser.add_argument(
        '--out-csv',
        type=str,
        default="_output/hyperparam_res.csv")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    all_calib_res = pd.read_csv(args.scores_file)
    all_calib_res = all_calib_res.drop(["time", "subject_i"], axis=1)
    with open(args.recalibrators_file, "rb") as f:
        recalib_dict = pickle.load(f)

    all_calib_res = all_calib_res.groupby(["mdl", "pop_idx", "measure"]).mean().reset_index()
    print(all_calib_res)
    all_calib_res.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
