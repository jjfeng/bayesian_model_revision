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
    parser = argparse.ArgumentParser(description='Summarize csvs by taking mean')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='seed')
    parser.add_argument(
        '--pivot-cols',
        type=str,
        help="pivot cols")
    parser.add_argument(
        '--pivot-rows',
        type=str,
        help="pivot rows")
    parser.add_argument(
        '--measure-filter',
        type=str,
        help="measure to summarize",
        default="nll")
    parser.add_argument(
        '--pop-idx-filter',
        type=str,
        help="pop to summarize",
        default="All")
    parser.add_argument(
        '--id-cols',
        type=str,
        help="columns to use as ids, remaining will be treated as values to summarize")
    parser.add_argument(
        '--value-col',
        type=str,
        help="value to plot")
    parser.add_argument(
        '--results',
        type=str,
        default="_output/calib_scores.csv")
    #parser.add_argument(
    #    '--out-tex',
    #    type=str,
    #    default="_output/hyperparam_res.tex")
    parser.add_argument(
        '--out-csv',
        type=str,
        default="_output/hyperparam_res.csv")
    parser.add_argument(
        '--out-fig',
        type=str,
        default=None)
    args = parser.parse_args()
    args.pop_idx_filter = args.pop_idx_filter.split(",")
    args.measure_filter = args.measure_filter.split(",")
    args.pivot_rows = args.pivot_rows.split(",")
    args.pivot_cols = args.pivot_cols.split(",")
    args.results = args.results.split(",")
    args.id_cols = args.id_cols.split(",")
    return args

def main():
    args = parse_args()
    logging.info(args)
    logging.info("Number of replicates: %d", len(args.results))

    all_res = []
    for idx, res_file in enumerate(args.results):
        if os.path.exists(res_file):
            res = pd.read_csv(res_file)
            all_res.append(res)
        else:
            print("file missing", res_file)
    num_replicates = len(all_res)
    all_res = pd.concat(all_res)
    all_res_mean = all_res.groupby(args.id_cols).mean().reset_index()
    all_res_std = (all_res.groupby(args.id_cols).std()/np.sqrt(num_replicates)).reset_index()
    all_res_std["zagg"] = "se"
    all_res_mean["zagg"] = "mean"
    all_res = pd.concat([all_res_mean, all_res_std]).sort_values(["measure", "pop_idx", "zagg", "mdl"])
    mask = all_res.pop_idx.isin(args.pop_idx_filter) & all_res.measure.isin(args.measure_filter)
    out_df = all_res[mask].pivot(args.pivot_rows, args.pivot_cols + ["zagg"], ["value"])
    marblr_key = [k for k in out_df.index.tolist() if k.startswith("marBLR")][0]
    out_df = out_df.rename(index={marblr_key: "MarBLR"})
    out_df = out_df.reindex(["MarBLR", "BLR", "adam", "cumulativeLR", "locked"])

    # Format so that we get mean (standard error) in each cell
    for col1 in out_df.columns.get_level_values(1).unique():
        for col2 in out_df.columns.get_level_values(2).unique():
            slice_df12 = out_df.iloc[:, (out_df.columns.get_level_values(1)==col1) & (out_df.columns.get_level_values(2)==col2)]
            col_select = ((out_df.columns.get_level_values(1)==col1) & (out_df.columns.get_level_values(2)==col2) & (out_df.columns.get_level_values(3)=="mean"))
            col_select = np.where(col_select)[0]
            out_df.iloc[:, col_select] = slice_df12.iloc[:,slice_df12.columns.get_level_values(3) == 'mean'].iloc[:,0].map('{:,.3f}'.format) + " (" + slice_df12.iloc[:,slice_df12.columns.get_level_values(3) == 'se'].iloc[:,0].map('{:,.3f}'.format) + ")"
    out_df = out_df.iloc[:,out_df.columns.get_level_values(3)=="mean"]
    print(out_df)

    with open(args.out_csv, "w") as f:
        f.writelines(out_df.to_csv(index=True, float_format="%.3f"))
    #all_res.to_csv(args.out_csv, index=False)
    #print(all_res)

    if args.out_fig:
        all_res_pivot = all_res.pivot(args.id_cols[0], args.id_cols[1], args.value_col)
        print(all_res_pivot)
        ax = sns.heatmap(all_res_pivot, cmap="YlGnBu")
        plt.savefig(args.out_fig)

if __name__ == "__main__":
    main()

