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

def get_hue_order(calib_res):
    hue_dict = {
            "Locked": "gray",
            "BLR": "midnightblue",
            "CumulativeLR": "red",
            "Adam": "orange",
            "mccormick": "skyblue",
            "regret_locked": "brown",
            "regret_dynamic": "blue",
            }
    mdl_strs = np.unique(calib_res.mdl).tolist()
    marBLR_keys = [mdl_str for mdl_str in mdl_strs if mdl_str.startswith("MarBLR")]
    other_keys = list(sorted([mdl_str for mdl_str in mdl_strs if (mdl_str in hue_dict.keys() and mdl_str != "BLR")]))
    hue_order = ["BLR"] + marBLR_keys + other_keys
    print(hue_order)
    if len(marBLR_keys) == 1:
        hue_dict[marBLR_keys[0]] = "lime"

    return hue_order, hue_dict

def plot_calib_scores_by_single_measure(calib_res, out_file, plot_legend=True):
    measure = calib_res.measure.unique()[0].upper()
    mdl_strs = np.unique(calib_res.mdl).tolist()
    marBLR_key = [mdl_str for mdl_str in mdl_strs if
            mdl_str.startswith("marBLR")][0]
    calib_res = calib_res.replace({'mdl': {marBLR_key: "MarBLR"}})
    hue_order, hue_dict = get_hue_order(calib_res)
    plt.clf()
    print(calib_res[calib_res["mdl"].isin(list(hue_dict.keys()))].mdl.unique())
    g = sns.FacetGrid(
            calib_res[calib_res["mdl"].isin(list(hue_dict.keys()))],
            row="pop_idx",
            sharex=True,
            sharey="col",
            )
    g.map_dataframe(sns.lineplot, x="time", y="value", style="mdl", hue="mdl", hue_order=hue_order, style_order=hue_order, palette=hue_dict)
    g.set_axis_labels("Time", measure)
    if plot_legend:
        g.add_legend(loc="center right")
    g.set_titles('')
    plt.tight_layout()
    plt.savefig(out_file)

def plot_calib_scores_by_measure(calib_res, out_file, plot_legend=True):
    mdl_strs = np.unique(calib_res.mdl).tolist()
    marBLR_key = [mdl_str for mdl_str in mdl_strs if
            mdl_str.startswith("marBLR")][0]
    calib_res = calib_res.replace({'mdl': {marBLR_key: "MarBLR", "locked": "Locked", "adam": "Adam", "cumulativeLR": "CumulativeLR"},
        "measure": {"eci": "ECI", "nll": "NLL", "auc": "AUC"}})
    hue_order, hue_dict = get_hue_order(calib_res)
    plt.clf()
    print(calib_res[calib_res["mdl"].isin(list(hue_dict.keys()))].mdl.unique())
    g = sns.FacetGrid(
            calib_res[calib_res["mdl"].isin(list(hue_dict.keys()))],
            col="measure",
            row="pop_idx",
            sharex=True,
            sharey="col",
            )
    g.map_dataframe(sns.lineplot, x="time", y="value", style="mdl", hue="mdl", hue_order=hue_order, style_order=hue_order, palette=hue_dict)
    if plot_legend:
        g.add_legend()
    g.set_titles('{col_name}')
    g.set_axis_labels("Time", "")
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(out_file)

def parse_args():
    parser = argparse.ArgumentParser(description='run simulation')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='seed')
    parser.add_argument(
        '--subpops',
        type=str,
        default="All",
        help='which subpops to plot')
    parser.add_argument(
        '--show-legend',
        default=False,
        action="store_true")
    parser.add_argument(
        '--plot-measures',
        type=str,
        default="nll")
    parser.add_argument(
        '--scores-file',
        type=str,
        default="_output/calib_scores.csv")
    parser.add_argument(
        '--obs-scores-file',
        type=str,
        default="_output/obs_scores.csv")
    parser.add_argument(
        '--recalibrators-file',
        type=str,
        default="_output/recalibrators.pkl",
        help='history recalib file')
    parser.add_argument(
        '--out-fig',
        type=str,
        default="_output/eci.png")
    parser.add_argument(
        '--out-regret-fig',
        type=str,
        default="_output/regret_obs.png")
    args = parser.parse_args()
    args.plot_measures = args.plot_measures.split(",")
    args.subpops = args.subpops.split(",")
    return args

def main():
    args = parse_args()

    all_calib_res = pd.read_csv(args.scores_file)
    with open(args.recalibrators_file, "rb") as f:
        recalib_dict = pickle.load(f)

    print(args.out_fig)
    all_calib_res = all_calib_res[all_calib_res["measure"].isin(args.plot_measures)]
    print(all_calib_res)
    all_calib_res = all_calib_res[all_calib_res["pop_idx"].isin(args.subpops)]
    tot_time = all_calib_res["time"].unique().size

    # Plot ECI and NLL instataneously
    sns.set_context("paper", font_scale=1.3)
    if len(args.plot_measures) == 1:
        plot_calib_scores_by_single_measure(all_calib_res, out_file=args.out_fig,
            plot_legend=args.show_legend)
    else:
        plot_calib_scores_by_measure(all_calib_res, out_file=args.out_fig,
            plot_legend=args.show_legend)


if __name__ == "__main__":
    main()
