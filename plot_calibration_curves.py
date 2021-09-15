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

def const_line(*args, **kwargs):
    x = np.array([0,1])
    y = np.array([0,1])
    plt.plot(y, x, linestyle='--')

def plot_calibration_curves(history, p1_name, p2_name, out_fig, time_mod = 1, legend=False):
    history_filter = history[history.time % time_mod == 0]
    history_filter["Time"] = history_filter.time + 1
    plt.clf()
    g = sns.lmplot(x=p1_name, y = p2_name, col="mdl", row="pop_idx", hue="Time", data=history_filter, lowess=True, scatter=False, palette="flare", truncate=True,legend=legend)
    g = g.set(xlim=(0, 1), ylim=(0, 1))
    g.map(const_line)
    g.set_titles('{col_name}')
    plt.ylabel("Predicted probability")
    g.set_axis_labels('Predicted', 'Observed')
    plt.tight_layout()
    plt.savefig(out_fig)

def parse_args():
    parser = argparse.ArgumentParser(description='plot calibration curves')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='seed')
    parser.add_argument(
        '--show-legend',
        default=False,
        action='store_true')
    parser.add_argument(
        '--sample-rate',
        type=float,
        default=1,
        help='points to sample')
    parser.add_argument(
        '--time-mod',
        type=int,
        default=1,
        help='plot times')
    parser.add_argument(
        '--history-file',
        type=str,
        default="_output/history.csv")
    #parser.add_argument(
    #    '--out-estimated-fig',
    #    type=str,
    #    default="_output/calib_curves_estim.png")
    parser.add_argument(
        '--out-errors-fig',
        type=str,
        default="_output/calib_curves_err.png")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    sns.set_context("paper", font_scale=2.3)

    history = pd.read_csv(args.history_file)
    oracle_mdls = [mdl_name for mdl_name in history.mdl.unique() if mdl_name.startswith("oracle")]
    history = history[~history["mdl"].isin(oracle_mdls)]
    marBLR_key = [k for k in history.mdl.unique() if k.startswith("marBLR")][0]
    print(marBLR_key)
    history = history.replace({"mdl": {
        marBLR_key: "MarBLR",
        "locked": "Locked",
        "adam": "Adam",
        "cumulativeLR": "CumulativeLR"
        }})

    # Accelerate plotting by sampling points
    keep_rvs = np.random.choice(2, size=history.shape[0], p=[1 - args.sample_rate, args.sample_rate])
    history["keep"] = keep_rvs
    history = history[history.keep == 1]

    plot_calibration_curves(history, "recalib_p", "true_mu", args.out_errors_fig, time_mod=args.time_mod, legend=args.show_legend)

    #history = history[(history["mdl"] != "locked") & (history["mdl"] != "bayes")]
    #print(history)
    #plot_calibration_curves(history, "orig_p", "recalib_p", args.out_estimated_fig, time_mod=args.time_mod, legend=args.show_legend)


if __name__ == "__main__":
    main()

