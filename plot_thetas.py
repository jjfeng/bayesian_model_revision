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
from matplotlib.lines import Line2D

def parse_args():
    parser = argparse.ArgumentParser(description='plot theta varying over time')
    parser.add_argument(
        '--recalibrators-file',
        type=str,
        default="_output/history.csv")
    parser.add_argument(
        '--plot-recalibs',
        type=str,
        default="BLR")
    parser.add_argument(
        '--do-rotate',
        default=False,
        action="store_true")
    parser.add_argument(
        '--show-legend',
        default=False,
        action="store_true")
    parser.add_argument(
        '--is-evolving-labels',
        default=False,
        action="store_true")
    parser.add_argument(
        '--time-mod',
        type=int,
        default=1,
        help="interval for plotting theta values")
    parser.add_argument(
        '--var-labels',
        type=str,
        default=None,
        help="labels for other variables")
    parser.add_argument(
        '--out-fig-time',
        type=str,
        default="_output/thetas_time.png")
    parser.add_argument(
        '--out-fig-idx',
        type=str,
        default="_output/thetas_idx.png")
    args = parser.parse_args()
    args.plot_recalibs = args.plot_recalibs.split(",")
    offset_var = 3 if args.is_evolving_labels else 2
    args.var_labels_dict = {i + offset_var: lab for i,lab in enumerate(args.var_labels.split(","))} if args.var_labels is not None else {}
    return args

def main():
    args = parse_args()

    with open(args.recalibrators_file, "rb") as f:
        recalibrators_dict = pickle.load(f)

    all_theta_dfs = []
    for k in recalibrators_dict.keys():
        if not any([k.startswith(m) for m in args.plot_recalibs]):
            continue
        print("reviser", k)

        recalib_mdl = recalibrators_dict[k]

        # Collect all thetas from fixedshare
        # assemble into dataframe
        thetas = np.vstack(recalib_mdl.theta_hist)
        theta_df = pd.DataFrame(thetas, columns=["Theta%d" % i for i in range(thetas.shape[1])])
        # The first timestamp is a dummy timestamp, just set to one day before
        theta_df["Time"] = np.array([recalib_mdl.timestamps[0] - 1] + recalib_mdl.timestamps) if recalib_mdl.timestamps[0] is not None else np.arange(thetas.shape[0])
        theta_df["Time_idx"] = np.arange(thetas.shape[0])
        theta_df = pd.wide_to_long(theta_df, stubnames="Theta", i="Time_idx", j="Parameter").reset_index()
        if args.is_evolving_labels:
            theta_df = theta_df.replace({'Parameter': {0: "Intercept", 1: "Original model", 2: "Evolving model"}})
        else:
            theta_df = theta_df.replace({'Parameter': {0: "Intercept", 1: "Original model"}})
        theta_df = theta_df.replace({'Parameter': args.var_labels_dict})
        theta_df["Reviser"] = k.split("_")[0].replace("mar", "Mar")
        print(theta_df)
        all_theta_dfs.append(theta_df)
    all_theta_dfs = pd.concat(all_theta_dfs)

    sns.set_context("paper", font_scale=2.4)
    # plot with the raw idxs
    #plt.clf()
    #g = sns.lineplot(x="Time_idx", y="Theta", hue="Parameter", style="Reviser",
    #        data=all_theta_dfs, legend=args.show_legend)
    #plt.title("Model revision parameters")
    #plt.ylabel("")
    #sns.despine()
    #plt.savefig(args.out_fig_idx, bbox_inches='tight')


    # plot with actual times
    print("begin plot")
    plt.clf()
    all_theta_dfs = all_theta_dfs[all_theta_dfs.Time_idx % args.time_mod ==0]
    if args.show_legend:
        ax = sns.lineplot(x="Time", y="Theta", hue="Parameter", style="Reviser",
            data=all_theta_dfs)
        legend = ax.legend()
        handles = []
        for hdl in legend.legendHandles:
            if not hdl._visible:
                continue
            if hdl._label in ["BLR", "MarBLR"]:
                continue
            hdl_blr = Line2D([0],[0],color=hdl.get_color(), linestyle="-", label=hdl._label + "+BLR")
            handles.append(hdl_blr)
        for hdl in legend.legendHandles:
            if not hdl._visible:
                continue
            if hdl._label in ["BLR", "MarBLR"]:
                continue
            hdl_marblr = Line2D([0],[0],color=hdl.get_color(), linestyle="--", label=hdl._label + "+MarBLR")
            handles.append(hdl_marblr)
        ax.legend(handles=handles, bbox_to_anchor=(-1.1, 1), loc=2, borderaxespad=0., framealpha=0)
    else:
        sns.lineplot(x="Time", y="Theta", hue="Parameter", style="Reviser",
                data=all_theta_dfs, legend=False)
    if args.do_rotate:
        plt.xticks(rotation=45)
    plt.title("Model revision parameters")
    plt.ylabel("")
    sns.despine()
    plt.savefig(args.out_fig_time, bbox_inches='tight')

if __name__ == "__main__":
    main()

