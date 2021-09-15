#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, plot_roc_curve, roc_curve
import seaborn as sns
from matplotlib import pyplot as plt

from dataset import make_safe_prob
N_BOOTSTRAPS = 200
COLOR_DICT = {"Locked": "gray", "BLR":"midnightblue", "MarBLR": "lime", "adam": "pink", "Ideal": "orange", "cumulativeLR": "purple", "Evolving-only": "orchid"}
LINE_DICT = {"Locked": "--", "BLR":"-.", "MarBLR": "--", "adam": "dotted", "cumulativeLR": "dotted", "Evolving-only": "dotted"}

def plot_calibration_curves(history, p1_name, p2_name, out_fig):
    if out_fig is None:
        print("no calib plot")
        return
    print("MEAN", np.sum(history.y[history["mdl"] == "adam"]))
    history["time_period"] = history.time_period + 1
    plt.clf()
    g = sns.catplot(
            x=p1_name,
            y = p2_name,
            hue="mdl",
            data=history,
            kind="point",
            col="time_period",
            ci=None,
            linestyles="--",
            legend=False,
            palette=COLOR_DICT)
    g.set_titles('Time period {col_name}')
    g.set(xticklabels=[0,"",0.2,"",0.4,"",0.6,"",0.8,"",1.0])
    #g.legend_.set_title(None)
    g.set_axis_labels("Predicted", "Observed")
    plt.subplots_adjust(bottom=0.2, left=0.06)
    plt.savefig(out_fig)

def plot_nll(history, out_fig):
    if out_fig is None:
        print("no nll plot")
        return
    history["nll_avg"] = history["nll"]
    for mdl_name in history.mdl.unique():
        mask = history["mdl"] == mdl_name
        history["nll_avg"][mask] = history[mask].nll.expanding().mean()
    plt.clf()
    sns.lineplot(x="time", y="nll_avg", style="mdl", hue="mdl", data=history)
    #, hue_order=hue_order, style_order=hue_order, palette=hue_dict)
    plt.savefig(out_fig)

def parse_args():
    parser = argparse.ArgumentParser(description='plot calibration curves for observed data')
    parser.add_argument(
        '--num-batches',
        type=int,
        default=1,
        help='number of observations to group together as a time batch')
    parser.add_argument(
        '--history-file',
        type=str,
        default="_output/history.csv")
    parser.add_argument(
        '--other-history-file',
        type=str,
        default=None)
    parser.add_argument(
        '--other-mdl',
        type=str)
    parser.add_argument(
        '--out-nll-fig',
        type=str)
    parser.add_argument(
        '--out-roc-fig',
        type=str)
    parser.add_argument(
        '--out-errors-fig',
        type=str)
    parser.add_argument(
        '--out-csv',
        type=str,
        default="_output/curve_err.csv")
    parser.add_argument(
        '--mdls',
        type=str,
        default="Locked,BLR,MarBLR")
    parser.add_argument(
        '--log-file',
        type=str,
        default="_output/log.txt")
    args = parser.parse_args()
    args.mdls = args.mdls.split(",")
    return args

def get_nll(pred_mu, true_y):
    pred_mu = make_safe_prob(pred_mu)
    log_liks = np.log(pred_mu) * true_y + np.log(1 - pred_mu) * (1 - true_y)
    return -log_liks

def get_eci_std_err2(df):
    bootstrap_res = []
    for i in range(N_BOOTSTRAPS):
        indices = np.random.randint(0, df.shape[0], df.shape[0])
        df_bootstrap = df.iloc[indices]
        eci_boot = get_eci(df_bootstrap)
        bootstrap_res.append(eci_boot)
    return np.var(bootstrap_res)

def get_eci(df):
    poly = PolynomialFeatures(degree=4, interaction_only=False)
    y_true = df.y.to_numpy().ravel()
    y_pred = df.recalib_p.to_numpy().reshape((-1,1))
    new_pred_logit = poly.fit_transform(np.log(y_pred/(1 - y_pred)))
    poly_reg = LogisticRegression().fit(new_pred_logit, y_true)
    calibration_curve_fit = poly_reg.predict(new_pred_logit)
    eci = np.mean(np.power(calibration_curve_fit.flatten() - y_pred.flatten(), 2)) * 100
    return eci

def get_auc_std_err2(df):
    """
    @return AUC std err via bootstrap
    """
    y_true = df.y.ravel()
    y_pred = df.recalib_p.ravel()
    bootstrapped_scores = []
    for i in range(N_BOOTSTRAPS):
        # bootstrap by sampling with replacement on the prediction indices
        indices = np.random.randint(0, y_pred.size, y_pred.size)
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        #print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    return np.var(bootstrapped_scores)

def auc_group(df):
    return roc_auc_score(df.y, df.recalib_p)

def plot_roc(history, fig_name):
    if fig_name is None:
        print("no roc plot")
        return

    plt.clf()
    num_batches = history.time_period.unique().size
    fig, axs = plt.subplots(1, num_batches, figsize=(12, 3), sharey=True, sharex=True)
    fig.add_subplot(111, frameon=False)
    for time_period, ax in enumerate(axs):
        for recalib_name in history.mdl.unique():
            mask = (history.mdl == recalib_name) & (history.time_period == time_period)
            y_true = history.y[mask]
            mu_pred = history.recalib_p[mask]
            fpr_mdl, tpr_mdl, _ = roc_curve(y_true, mu_pred)
            ax.plot(fpr_mdl, tpr_mdl, label=recalib_name, linestyle=LINE_DICT[recalib_name], color=COLOR_DICT[recalib_name])
        if time_period == num_batches - 1:
            legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            legend.get_frame().set_linewidth(0)
    sns.despine()
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.tight_layout()
    plt.savefig(fig_name)

def main():
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)

    history = pd.read_csv(args.history_file)

    if args.other_history_file:
        other_history = pd.read_csv(args.other_history_file).replace({"mdl": {"locked": args.other_mdl}})
        history = pd.concat([history,other_history])

    history.time = pd.to_datetime(history.time)
    min_time = np.min(history.time)
    batch_size = np.ceil((np.max(history.time) - min_time).days/args.num_batches) + args.num_batches
    history["time_period"] = np.floor((history.time - min_time).dt.days/batch_size).astype(int)
    #history["brier_diff"] = (history.recalib_p - history.y)**2
    history["nll"] = get_nll(history.recalib_p.to_numpy(), history.y.to_numpy())
    marBLR_keys = [mdl_str for mdl_str in history.mdl.unique() if mdl_str.startswith("marBLR")]
    if marBLR_keys:
        history = history.replace({"mdl": {marBLR_keys[0]: "MarBLR", "locked": "Locked"}})
    else:
        history = history.replace({"mdl": {"locked": "Locked"}})
    if args.mdls:
        history = history[history.mdl.isin(args.mdls)]

    # Plotting
    sns.set_context("paper", font_scale=1.5)
    plot_roc(history, args.out_roc_fig)

    # Group by rounding
    sns.set_context("paper", font_scale=2.3)
    history["Predicted"] = np.round(history.recalib_p * 10).astype(int)/10
    print(history.Predicted)
    ideal_hist = pd.DataFrame({"Predicted": np.arange(11)/10, "y": np.concatenate([[0.025], np.arange(1,10)/10,[0.975]])})
    ideal_hist["mdl"] = "Ideal"
    new_hist = [history[["time_period", "mdl", "Predicted", "y"]]]
    for time_b in history.time_period.unique():
        ideal_hist_new = ideal_hist.copy()
        ideal_hist_new["time_period"] = time_b
        new_hist.append(ideal_hist_new)
    plot_calibration_curves(pd.concat(new_hist), "Predicted", "y", args.out_errors_fig)

    print("NLL")
    nll_res = history[["nll", "mdl"]].groupby("mdl").agg(["mean", "var"]).reset_index()
    nll_res.columns = nll_res.columns.droplevel()
    nll_res["var"] /= np.max(history.subject_i + 1)
    nll_res = nll_res.rename(columns={"": "mdl", "mean": "nll", "var": "nll_std_err2"})
    nll_res["nll_ci_lower"] = nll_res["nll"] - 1.96 * np.sqrt(nll_res["nll_std_err2"])
    nll_res["nll_ci_upper"] = nll_res["nll"] + 1.96 * np.sqrt(nll_res["nll_std_err2"])
    nll_res = nll_res.drop(["nll_std_err2"], axis=1)
    print(nll_res)

    # ECI
    print("ECI")
    eci_batch = history.groupby(["time_period", "mdl"]).apply(get_eci).reset_index()
    eci_batch_se2 = history.groupby(["time_period", "mdl"]).apply(get_eci_std_err2).reset_index()
    eci_sum = eci_batch.merge(eci_batch_se2, on=["time_period", "mdl"]).rename(columns={"0_x": "eci", "0_y": "eci_std_err2"})
    logging.info("ECI")
    logging.info(eci_sum)
    eci_avgs = eci_sum.groupby(["mdl"]).mean().reset_index()
    eci_avgs["eci_ci_lower"] = eci_avgs["eci"] - 1.96 * np.sqrt(eci_avgs["eci_std_err2"])
    eci_avgs["eci_ci_upper"] = eci_avgs["eci"] + 1.96 * np.sqrt(eci_avgs["eci_std_err2"])
    eci_avgs = eci_avgs.drop(["time_period", "eci_std_err2"], axis=1)
    print(eci_avgs)

    # AUC
    print("AUC")
    auc_batch = history.groupby(["time_period", "mdl"]).apply(auc_group).reset_index()
    auc_batch_se2 = history.groupby(["time_period", "mdl"]).apply(get_auc_std_err2).reset_index()
    auc_sum = auc_batch.merge(auc_batch_se2, on=["time_period", "mdl"]).rename(columns={"0_x": "auc", "0_y": "auc_std_err2"})
    logging.info("AUC")
    logging.info(auc_sum)
    auc_avgs = auc_sum.groupby(["mdl"]).mean().reset_index()
    auc_avgs["auc_ci_lower"] = auc_avgs["auc"] - 1.96 * np.sqrt(auc_avgs["auc_std_err2"])
    auc_avgs["auc_ci_upper"] = auc_avgs["auc"] + 1.96 * np.sqrt(auc_avgs["auc_std_err2"])
    auc_avgs = auc_avgs.drop(["time_period", "auc_std_err2"], axis=1)
    print(auc_avgs)

    grouped_ps = auc_avgs.merge(eci_avgs, on=["mdl"]).merge(nll_res, on=["mdl"])

    # Calculate mean ECI, NLL, and AUC
    print(grouped_ps)
    auc_strs = []
    eci_strs = []
    nll_strs = []
    for idx in range(grouped_ps.shape[0]):
        auc_strs.append("%.3f (%.3f,%.3f)" % (grouped_ps.auc[idx], grouped_ps.auc_ci_lower[idx], grouped_ps.auc_ci_upper[idx]))
        eci_strs.append("%.3f (%.3f,%.3f)" % (grouped_ps.eci[idx], grouped_ps.eci_ci_lower[idx], grouped_ps.eci_ci_upper[idx]))
        nll_strs.append("%.3f (%.3f,%.3f)" % (grouped_ps.nll[idx], grouped_ps.nll_ci_lower[idx], grouped_ps.nll_ci_upper[idx]))
    out_df = pd.DataFrame({"mdl": grouped_ps.mdl, "auc": auc_strs, "eci": eci_strs, "nll": nll_strs})
    print(out_df)
    with open(args.out_csv, "w") as f:
        out_df.to_csv(f)

    plot_nll(history, args.out_nll_fig)



if __name__ == "__main__":
    main()

