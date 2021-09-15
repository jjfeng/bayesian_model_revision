#!/usr/bin/env python
"""
Main file for running model revisers online on a given data stream
Runs MarBLR and BLR, compares against other logistic model revisers
"""

import sys, os
import time
import argparse
import pickle
import logging
import progressbar
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, plot_roc_curve, roc_curve

from embedding import EmbeddingMaker, SubgroupsEmbeddingMaker
from recalibrator import *
from dataset import *

def get_obs_score(pred_mu, true_y, true_mu:float = None):
    pred_mu = make_safe_prob(pred_mu)
    if true_mu is not None and (true_mu[0,0] is not None):
    # negative log lik, take into account importance sampling weights
        log_liks = np.log(pred_mu) * true_mu + np.log(1 - pred_mu) * (1 - true_mu)
    else:
        log_liks = np.log(pred_mu) * true_y + np.log(1 - pred_mu) * (1 - true_y)
    nll = -np.mean(log_liks)

    return pd.DataFrame({
        "measure":["nll"],
        "value":[nll]
    })

def get_calibration_score(pred_mu, dataset):
    """
    Calculates the true scores assuming true probabilities are available
    """
    # Polynomial regression for ECI
    poly = PolynomialFeatures(degree=4, interaction_only=False)
    new_pred_mu = poly.fit_transform(pred_mu)
    poly_reg = LinearRegression().fit(new_pred_mu, dataset.mu.flatten())
    calibration_curve_fit = poly_reg.predict(new_pred_mu)
    eci = np.mean(np.power(calibration_curve_fit.flatten() - pred_mu.flatten(), 2)) * 100

    # negative log lik, take into account importance sampling weights
    log_liks = np.log(pred_mu) * dataset.mu + np.log(1 - pred_mu) * (1 - dataset.mu)
    assert (pred_mu.shape == dataset.mu.shape)
    if dataset.weight is not None:
        nll = -np.sum(log_liks * dataset.weight)/np.sum(dataset.weight)
    else:
        nll = -np.mean(log_liks)

    auc = roc_auc_score(dataset.y, pred_mu)

    return pd.DataFrame({
        "measure":["eci", "nll", "auc"],
        "value":[eci, nll, auc]
        })

def get_calibration_score_all(pred_mus, datasets, proportions, t_idx, recalib_name):
    all_dfs = []
    for pop_idx, (pred_mu, dataset) in enumerate(zip(pred_mus, datasets)):
        df = get_calibration_score(pred_mu, dataset)
        df["pop_idx"] = str(pop_idx)
        all_dfs.append(df)
    # Get calibration score on all
    merge_df = get_calibration_score(np.concatenate(pred_mus), Dataset.merge(datasets, proportions))
    merge_df["pop_idx"] = "All"
    return pd.concat(all_dfs + [merge_df])

def do_model_predict_probs(modeler, newx):
    return modeler.predict_prob_single(newx).reshape((-1,1))


def parse_args():
    parser = argparse.ArgumentParser(description='run simulation')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='seed')
    parser.add_argument(
        '--embedding-idxs',
        type=str,
        help="comma-separated list of other variable indices to pass to the recalibrator")
    parser.add_argument(
        '--is-subgroup-embedding',
        default=False,
        action="store_true",
        help="Embedding represents a patient subgroup? (of which there can only be two)")
    parser.add_argument(
        '--max-covariance-scale',
        type=float,
        default=1,
        help="max scaling factor for the prior covariance matrix to satisfy the Type I regret bounds")
    parser.add_argument(
        '--inflation-rates',
        type=str,
        help="comma-separated inflation rates for the prior covariance matrix to satisfy Type I regret bounds")
    parser.add_argument(
        '--alphas',
        type=str,
        help="comma-separated alpha transition probs to test (we will pick one to satisfy Type I regret bounds)")
    parser.add_argument(
        '--type-i-regret-factor',
        type=float,
        default=0.15,
        help="type I regret factor control for BLR prior")
    parser.add_argument(
        '--obs-batch-size',
        type=int,
        default=1,
        help="number of observations per batch for updating the model reviser and underlying models")
    parser.add_argument(
        '--test-batch',
        type=int,
        default=1,
        help="number of observations per batch for testing the performance of the model reviser on provided test data")
    parser.add_argument(
        '--hist-batch',
        type=int,
        default=2,
        help="number of observations per batch to log results")
    parser.add_argument(
        '--reference-recalibs',
        type=str,
        default="locked,adam,cumulativeLR",
        help="comma-separate recalibrators to test against")
    parser.add_argument(
        '--data-file',
        type=str,
        default="_output/data.pkl",
        help="input data file")
    parser.add_argument(
        '--model-file',
        type=str,
        default="_output/models.pkl",
        help="input file with underlying models")
    parser.add_argument(
        '--history-file',
        type=str,
        default="_output/history.csv",
        help="output file with recorded model reviser performance")
    parser.add_argument(
        '--scores-file',
        type=str,
        default="_output/scores.csv",
        help="output file with scores on the test data")
    parser.add_argument(
        '--obs-scores-file',
        type=str,
        default="_output/obs_scores.csv",
        help="output file with scores on the observed data")
    parser.add_argument(
        '--recalibrators-file',
        type=str,
        default="_output/recalibrators.pkl",
        help="output file with trained model revisers")
    parser.add_argument(
        '--log-file',
        type=str,
        default="_output/log.txt")
    args = parser.parse_args()
    # We only accept a simple basis into the model revisers (just take in the model scores directly, no additional transformations of the model scores)
    args.basis = 1
    args.embedding_idxs = list(map(int, args.embedding_idxs.split(","))) if len(args.embedding_idxs) else []
    args.alphas = list(map(float, args.alphas.split(","))) if args.alphas is not None else []
    args.inflation_rates = list(map(float, args.inflation_rates.split(","))) if args.inflation_rates is not None else []
    args.reference_recalibs = args.reference_recalibs.split(",")
    return args

def make_logistic_reg():
    return LogisticRegression(penalty="none", solver="lbfgs", warm_start=True)

def make_reference_recalibrators(recalibs: List[str], tot_time: int, basis: int = 1) -> Dict[str, Recalibrator]:
    # create all recalibration models first
    recalib_locked = LockedRecalibrator(make_logistic_reg(), basis=basis)
    recalib_adam = AdamRecalibrator(make_logistic_reg(), eta=0.01)
    recalib_cum = CumulativeRecalibrator(make_logistic_reg(), basis=basis)
    recalibrators = {
            "locked": recalib_locked,
            "adam": recalib_adam,
            "cumulativeLR": recalib_cum,
            "oracle_locked": OracleRecalibrator(basis=basis),
            "oracle_dynamic": OracleRecalibrator(basis=basis),
    }

    # Return only the selected recalibrators
    selected_recalibrators = {recalib_key: recalibrators[recalib_key] for recalib_key in recalibs}
    return selected_recalibrators

def main():
    args = parse_args()
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)
    # parameters
    np.random.seed(args.seed)
    logging.info(args)

    with open(args.data_file, "rb") as f:
        data = pickle.load(f)["full_dat"]
        tot_time = data.size
        init_recalib_data = data.get_init_recalib_dat()
    print("data done")

    with open(args.model_file, "rb") as f:
        modeler = pickle.load(f)
    print("modeler done")

    if args.is_subgroup_embedding:
        emb_maker = SubgroupsEmbeddingMaker(modeler, group_idxs = np.array(args.embedding_idxs))
    else:
        emb_maker = EmbeddingMaker(modeler, x_idxs = np.array(args.embedding_idxs))

    # create recalibration models
    recalibrators = make_reference_recalibrators(args.reference_recalibs, tot_time, basis=args.basis)

    for alpha in args.alphas:
        for inflation_rate in args.inflation_rates:
            recalib_fixedshare = MarBLRRecalibrator(make_logistic_reg(), inflation_rate=inflation_rate, alphas=[alpha, alpha], max_covariance_scale=args.max_covariance_scale, basis=args.basis)
            recalibrators[recalib_fixedshare.name] = recalib_fixedshare
    print("recalibrators created")

    # init recalibration models
    model_pred, locked_pred, _ = emb_maker.initialize(init_recalib_data.x)
    logging.info("Embedding max eigen %f", 1/model_pred.shape[0] * np.linalg.eigvalsh(model_pred.T @ model_pred).max())

    print("embeddings made")
    # Initialize with locked recalibrator
    recalibrators["locked"].init(locked_pred, init_recalib_data.y)
    test_dat = data.get_test_dats(0, do_merge=True)
    init_locked_score = recalibrators["locked"].init_score(emb_maker.embed(test_dat.x)[1], test_dat.y, test_dat.weight)
    print(recalibrators.keys())
    if len(list(recalibrators.keys())) > 1:
        remapped_locked_theta = emb_maker.remap_model_locked_to_full_params(recalibrators["locked"].init_theta)
    logging.info("Init locked score %f", init_locked_score)
    print("Init locked score %f", init_locked_score)

    # Initialize the other recalibrators  with the (remppaed) locked theta
    for recalib_name, recalib_mdl in recalibrators.items():
        print(recalib_name)
        if recalib_name == "locked":
            continue
        elif recalib_name.startswith("oracle"):
            test_dats = data.get_test_dats(None)
            test_model_preds_all = [emb_maker.embed(test_dat.x)[0] for pop_idx, test_dat in enumerate(test_dats)]
            if data.true_prob_avail:
                recalib_mdl.init(np.vstack(test_model_preds_all), np.vstack([test_dat.mu for test_dat in test_dats]))
            else:
                recalib_mdl.init(np.vstack(test_model_preds_all), np.vstack([test_dat.y for test_dat in test_dats]))
        elif recalib_name.startswith("marBLR") or recalib_name == "BLR":
            recalib_mdl.init(model_pred, init_recalib_data.y, init_theta=remapped_locked_theta, tot_time=tot_time/args.obs_batch_size, max_regret=init_locked_score * args.type_i_regret_factor, n=args.obs_batch_size)
        else:
            recalib_mdl.init(model_pred, init_recalib_data.y, init_theta=remapped_locked_theta)
        print("INIT THETA", recalib_mdl.init_theta)
    print("Done initializing recalibrators")

    # Run simulation
    all_calib_res = []
    all_obs_scores = []
    forecast_p_history = []
    obs_forecast_history = []
    did_refit = False
    bar = progressbar.ProgressBar(maxval=tot_time, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    last_subject = 0
    print("TOT TIME", tot_time)
    refit_batch_size = min(modeler.refit_freq if modeler.refit_freq is not None else tot_time, args.test_batch)
    for batch_start_i in range(0, tot_time + 1, refit_batch_size):
        bar.update(batch_start_i)

        # Evaluate on test data
        # If the test time is after the data stream is over, we just grab the last test batch, assuming the data is still IID
        test_dats = data.get_test_dats(batch_start_i if batch_start_i < tot_time else tot_time - 1, do_merge=False)
        # Do test things
        if data.true_prob_avail and (test_dats is not None) and ((batch_start_i == 0) or batch_start_i % args.test_batch == 0):
            if "oracle_dynamic" in recalibrators:
                # Construct the dynamic oracle model
                test_model_preds_all = [emb_maker.embed(test_dat.x)[0] for pop_idx, test_dat in enumerate(test_dats)]
                recalibrators["oracle_dynamic"].do_oracle_update(np.vstack(test_model_preds_all), np.vstack([test_dat.mu for test_dat in test_dats]))

            for recalib_name, recalib_mdl in recalibrators.items():
                forecast_ps = []
                oracle_ps = []
                for pop_idx, test_dat in enumerate(test_dats):
                    test_model_preds, locked_model_preds, test_model_probs = emb_maker.embed(test_dat.x)
                    recalib_p = recalib_mdl.predict(locked_model_preds if recalib_mdl.is_locked else test_model_preds)
                    forecast_ps.append(recalib_p)
                    if batch_start_i == 0 or ((batch_start_i + 1) % args.hist_batch == 0):
                        forecast_hist = pd.DataFrame({
                            "orig_p": test_model_probs.flatten(),
                            "recalib_p": recalib_p.flatten()})
                        if test_dat.mu is not None:
                            forecast_hist["true_mu"] = test_dat.mu.flatten()
                        forecast_hist["pop_idx"] = pop_idx
                        forecast_hist["time"] = data.get_timestamp(batch_start_i)
                        forecast_hist["subject_i"] = batch_start_i
                        forecast_hist["mdl"] = recalib_name
                        forecast_p_history.append(forecast_hist)

                recalib_scores = get_calibration_score_all(
                        forecast_ps,
                        test_dats,
                        data.proportions,
                        batch_start_i,
                        recalib_name)
                recalib_scores["time"] = data.get_timestamp(batch_start_i)
                recalib_scores["subject_i"] = batch_start_i
                recalib_scores["mdl"] = recalib_name
                #print("recalib", recalib_scores)
                logging.info(recalib_scores)
                all_calib_res.append(recalib_scores)

        if batch_start_i == tot_time:
            break

        # evaluate and then do recalibration update on new observation
        st_time = time.time()
        newx, newy, new_mu = data.get_obs(batch_start_i, batch_start_i + refit_batch_size)
        model_pred, locked_pred, orig_model_prob = emb_maker.embed(newx)
        for recalib_name, recalib_mdl in recalibrators.items():
            mdl_pred = locked_pred if recalib_mdl.is_locked else model_pred
            for idx in range(0, newy.size, args.obs_batch_size):
                subject_i = batch_start_i + idx
                mdl_pred_batch = mdl_pred[idx:idx + args.obs_batch_size]
                newy_batch = newy[idx:idx + args.obs_batch_size]
                new_mu_batch = new_mu[idx:idx + args.obs_batch_size]
                mini_batch_size = newy_batch.size

                recalib_mdl_pred = recalib_mdl.predict(mdl_pred_batch)
                obs_score = get_obs_score(recalib_mdl_pred, newy_batch, true_mu=new_mu_batch)
                obs_score["time"] = data.get_timestamp(subject_i)
                obs_score["subject_i"] = subject_i
                obs_score["mdl"] = recalib_name
                all_obs_scores.append(obs_score)

                obs_forecast_hist = pd.DataFrame({
                    "orig_p": orig_model_prob[idx:idx + args.obs_batch_size].flatten(),
                    "recalib_p": recalib_mdl_pred.flatten(),
                    "y": newy_batch.flatten(),
                    "subject_i": np.arange(subject_i, subject_i + mini_batch_size),
                })
                obs_forecast_hist["time"] = data.get_timestamp(subject_i)
                obs_forecast_hist["mdl"] = recalib_name
                obs_forecast_history.append(obs_forecast_hist)

                #num_repeat = 1 if mdl_pred[0,2] == 0  else int(np.random.rand() < 0.3)
                #for _ in range(num_repeat):
                recalib_mdl.update(mdl_pred, newy, did_refit, timestamp=data.get_timestamp(subject_i))

        # just logging some progress
        mean_nll = pd.concat(all_obs_scores, ignore_index=True).groupby(["mdl", "measure"]).mean()
        #print(mean_nll)
        logging.info(mean_nll)

        did_refit = modeler.update(newx, newy)
        print("REFIT", did_refit)

    bar.finish()

    # Plot and summarize
    if len(forecast_p_history) > 0:
        forecast_p_history = pd.concat(forecast_p_history, ignore_index=True)
        forecast_p_history.to_csv(args.history_file, index=False)

        all_calib_res = pd.concat(all_calib_res, ignore_index=True)
        all_calib_res.to_csv(args.scores_file, index=False)
        logging.info("TRUE EXPECTED SCORES")
        logging.info(all_calib_res.groupby(["mdl", "pop_idx", "measure"]).mean())

    all_obs_scores = pd.concat(all_obs_scores, ignore_index=True)
    mean_nll = all_obs_scores.groupby(["mdl", "measure"]).mean()
    logging.info("OBSERVED SCORES")
    logging.info(mean_nll)
    print(mean_nll)

    all_obs_history = pd.concat(obs_forecast_history, ignore_index=True)
    all_obs_history.to_csv(args.obs_scores_file, index=False)

    with open(args.recalibrators_file, "wb") as f:
        pickle.dump(recalibrators, f)

if __name__ == "__main__":
    main()
