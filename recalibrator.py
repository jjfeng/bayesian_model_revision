"""
Code for all the model revisers/recalibrators
"""
import sys, os
import logging
import time

import numpy as np
import scipy.stats
import scipy.special
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression

from logistic_regression import *
from regret_bounds import get_regret_bound_type_i


class LogisticPosterior:
    def __init__(self, theta, covariance):
        self.theta = theta.reshape((-1,1))
        self.d = theta.size
        self.covariance = covariance

    def __str__(self):
        return "Logistic Theta %s\nLogistic Cov\n%s" % (self.theta.flatten(), self.covariance)

    def _add_intercept(self, x: np.ndarray):
        return np.hstack([np.ones((x.shape[0], 1)), x])

    def _predict_map(self, x: np.ndarray, theta):
        """
        Predicts using MAP
        """
        x_aug = self._add_intercept(x)
        logit_preds = np.array(np.matmul(x_aug, theta), dtype=float)
        return 1/(1 + np.exp(-logit_preds))

    def predict_map(self, x: np.ndarray):
        """
        Predicts using MAP
        """
        x_aug = self._add_intercept(x)
        logit_preds = np.array(np.matmul(x_aug, self.theta), dtype=float)
        return 1/(1 + np.exp(-logit_preds))

    def predict(self, x: np.ndarray):
        """
        Does a laplace approximation
        """
        laplace_log_p = self.get_laplace_log_p_y(x, np.ones((x.shape[0], 1)))
        return np.exp(laplace_log_p)

    def _cvxpy_update(self,  x: np.ndarray, y: np.ndarray):
        """
        @return log E_theta[p(y|theta)] where theta is from the prior
        """
        assert y.size == 1

        x_aug = self._add_intercept(x)
        precision_mat = np.linalg.pinv(self.covariance)
        new_theta = solve_regularized_logistic(x_aug, y, self.theta, precision_mat)
        self.theta = new_theta.reshape((-1,1))
        y_hat = self.predict_map(x)

        self.covariance = np.linalg.pinv(y_hat * (1 - y_hat) * x_aug.T @ x_aug + precision_mat)
        return self.get_laplace_log_p_y(x, y)

    def get_laplace_log_p_y(self, x: np.ndarray, y: np.ndarray, eps: float=1e-10):
        y_hat = self.predict_map(x)
        log_p_theta = scipy.stats.multivariate_normal.logpdf(self.theta.ravel(), mean=self.theta.ravel(), cov=self.covariance)
        p_y = np.maximum(eps, np.minimum(1 - eps, self.predict_map(x)))
        log_p_obs_y = np.log(p_y) * y + np.log(1 - p_y) * (1 - y)
        return self.d/2 * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(self.covariance)) + log_p_theta + log_p_obs_y

    def update(self,  x: np.ndarray, y: np.ndarray):
        """
        update using newton step
        @return log E_theta[p(y|theta)] where theta is from the prior
        """
        y_hat = self.predict_map(x)
        x_aug = self._add_intercept(x)
        del1 = np.sum(((y - y_hat).T @ x_aug).T, axis=1, keepdims=True)
        del2 = -np.linalg.pinv(self.covariance) - x_aug.T @ np.diag(y_hat.ravel() * (1 - y_hat).ravel()) @ x_aug
        new_theta = self.theta - np.linalg.pinv(del2) @ del1
        self.theta = new_theta
        self.covariance = np.linalg.pinv(-del2)
        return np.sum([self.get_laplace_log_p_y(x[i:i + 1,:], y[i:i + 1,:]) for i in range(y.size)])

    def approx_mixture(self, other_logistic_posterior, weight: float = 0.5):
        assert (weight >= -1e-10)
        assert (weight <= 1 + 1e-10)
        new_theta = self.theta * weight + other_logistic_posterior.theta * (1 - weight)
        new_cov = self.covariance * weight + other_logistic_posterior.covariance * (1 - weight)
        return LogisticPosterior(new_theta, new_cov)


class ChangepointSeqPosterior:
    def __init__(self, changepoint_seq, logistic_post, scores=[]):
        self.changepoint_seq = changepoint_seq
        self.logistic_post = logistic_post
        self.scores = scores

    @property
    def len(self):
        return len(self.changepoint_seq)

    @property
    def num_change(self):
        return np.sum(self.changepoint_seq)

    def get_tot_weighted_subscores(self, num_back: int, alpha: float):
        num_change = np.sum(self.changepoint_seq[-num_back:])
        if (num_change > 0) and (num_back - num_change > 0):
            log_prior_prob = (num_back - num_change) * np.log(1 - alpha) + num_change * np.log(alpha)
        elif num_change == 0:
            log_prior_prob = num_back * np.log(1 - alpha)
        else:
            log_prior_prob = num_change * np.log(alpha)
        return np.sum(self.scores[-num_back:]) + log_prior_prob

    def predict(self, x: np.ndarray):
        return self.logistic_post.predict(x)

    def update(self,  x: np.ndarray, y: np.ndarray):
        """
        @return posterior predictive probability
        """
        update_score = self.logistic_post.update(x, y)
        self.scores.append(update_score)
        return update_score

    def extend_seq(self, do_change, inflation_rate, ref_cov: np.ndarray=None, max_inflat_rate: float = 0.01):
        """
        @param do_change: whether or not a shift occurs at this time
        @param inflation_rate: the inflation rate to apply to posterior covariance matrix if a shift occurs (otherwise do not inflate the posterior)
        @param ref_cov: this is the covariance matrix we are referring to for defining the inflation rate.
        @return a new posterior
        """
        # Inflate by either a multiple of the scaled covariance matrix where the maximum variance of any
        # parameter is no more than 1
        if (ref_cov is not None) and (inflation_rate > 0):
            ref_inflat_rate = inflation_rate * np.trace(ref_cov)/np.trace(self.logistic_post.covariance)
            inflation_rate = min(ref_inflat_rate, max_inflat_rate)
        inflat_mat = self.logistic_post.covariance * (inflation_rate if do_change else 0)

        change_covariance = self.logistic_post.covariance + inflat_mat

        new_logistic_posterior = LogisticPosterior(
                self.logistic_post.theta,
                change_covariance)

        return ChangepointSeqPosterior(
                self.changepoint_seq + [do_change],
                new_logistic_posterior,
                self.scores)


class Recalibrator:
    is_locked = False
    def __init__(self, logistic_reg, alpha: float = 0.01, num_back:int = 2, inflation_rate: float = 0.01, init_var: float = None, basis:int = 1):
        self.logistic_reg = logistic_reg
        self.alpha = alpha
        self.num_back = num_back
        self.inflation_rate = inflation_rate
        self.max_num_seqs = 2**num_back
        self.prefix_changept_seq = []
        self.change_options = [0,1] if self.alpha > 0 else [0]
        self.init_var = init_var
        self.theta_hist = []
        self.timestamps = []
        self.basis = basis

    def make_aug_x(self, x):
        if self.basis == 1:
            return x
        else:
            return np.hstack([x, np.maximum(x[:,0:1], 0)])

    def init(self, x, y, init_theta: np.ndarray = None):
        self.x = x
        self.y = y
        self.timepoints = [x.shape[0]]

        # Fit init logistic regression model. Set as prior fixed for now?
        aug_x = self.make_aug_x(x)
        if init_theta is None:
            self.logistic_reg.fit(aug_x, y.ravel())
            self.init_theta = np.concatenate([self.logistic_reg.intercept_, self.logistic_reg.coef_.ravel()])
        else:
            self.init_theta = init_theta

        init_covariance = get_logistic_cov(aug_x, y, self.init_theta.reshape((-1,1)))
        if self.init_var is not None:
            init_covariance = rescale_covariance(init_covariance, max_eig=self.init_var)
        init_posterior = LogisticPosterior(
                theta=self.init_theta,
                covariance=init_covariance)

        self.seq_posteriors = [ChangepointSeqPosterior([], init_posterior)]
        self.seq_weights = np.array([1])
        self.theta_hist.append(self.init_theta.ravel())

    def update(self, x, y, known_refit, timestamp = None):
        self.x = np.vstack([self.x, x])
        self.y = np.vstack([self.y, y])
        self.timepoints = self.timepoints + [self.timepoints[-1] + x.shape[0]]

        new_seq_posteriors = [[], []]
        all_seq_posteriors = []
        new_log_seq_scores = [[],[]]
        all_log_seq_scores = []
        for do_change in self.change_options:
            do_change_prob = self.alpha if do_change else (1 - self.alpha)
            for seq_post in self.seq_posteriors:
                new_seq_post = seq_post.extend_seq(do_change, self.inflation_rate)
                new_seq_post.update(self.make_aug_x(x), y)
                log_seq_score = new_seq_post.get_tot_weighted_subscores(self.num_back + 1, self.alpha)
                all_seq_posteriors.append(new_seq_post)
                if new_seq_post.len - self.num_back > 0:
                    seq_post_next = new_seq_post.changepoint_seq[new_seq_post.len - self.num_back - 1]
                    new_log_seq_scores[seq_post_next].append(log_seq_score)
                    new_seq_posteriors[seq_post_next].append(new_seq_post)
                all_log_seq_scores.append(log_seq_score)

        if self.alpha > 0  and (new_seq_post.len - self.num_back > 0):
            #print("change SCORES", logsumexp(new_log_seq_scores[0]), logsumexp(new_log_seq_scores[1]))
            change_append = 0 if logsumexp(new_log_seq_scores[0]) >= logsumexp(new_log_seq_scores[1]) else 1
            self.prefix_changept_seq += [change_append]

            # Select top scoring seqs, update weights
            self.seq_posteriors = new_seq_posteriors[change_append]
            # TODO: mask then normalize or normalize then mask?
            self.seq_weights = scipy.special.softmax(new_log_seq_scores[change_append])
        else:
            self.seq_posteriors = all_seq_posteriors
            self.seq_weights = scipy.special.softmax(all_log_seq_scores)
        self.theta_hist.append(self.get_mean_theta().ravel())
        self.timestamps.append(timestamp)

    def predict(self, x):
        predictions_raw = np.hstack([seq_post.predict(self.make_aug_x(x)) for seq_post in self.seq_posteriors])
        weighted_pred = predictions_raw * self.seq_weights.reshape((1,-1))
        predictions = np.sum(weighted_pred, axis=1, keepdims=True)
        return predictions

    def get_mean_theta(self):
        theta_raw = np.hstack([seq_post.logistic_post.theta for seq_post in self.seq_posteriors])
        weighted_theta = theta_raw * self.seq_weights.reshape((1,-1))
        mean_theta = np.sum(weighted_theta, axis=1, keepdims=True)
        return mean_theta

    def print_summary(self):
        print("prefix", self.prefix_changept_seq)
        print("poster theta", self.get_mean_theta().ravel())

class MarBLRRecalibrator(Recalibrator):
    """
    marBLR and BLR
    To run BLR, set inflation_rate = 0 (alpha rate will be ignored)
    """
    max_inflat_rate = 0.005

    def __init__(self, logistic_reg, alphas = [0.01, 0.01], inflation_rate: float = 0.01, init_var: float = None, basis: int = 1, max_covariance_scale: np.ndarray=1):
        """
        @param logistic_reg: the logistic regression model for doing the revision layer
        @param alphas: alphas[0] is the probability of doing a shift when the current time step had no shift, alphas[1] is the probability of doing a shift when the current time step had a shift
        @param inflation_rate: this is the inflation rate relative to the original prior covariance matrix
        @param init_var: initial prior covariance matrix if you want to set it manually, otherwise leave it None and
                    we will construct an initial prior covariance matrix using the initial recalibration data
                    and taking the standard error matrix
        @param basis: how many terms in the basis expansion (we only allow one right now)
        @param max_covariance_scale: this is the max inflation rate relative to the logistic posterior
a       """
        self.logistic_reg = logistic_reg
        self.alphas = alphas
        self.inflation_rate = inflation_rate
        self.change_options = [0,1] if self.alphas[0] > 0 else [0]
        self.init_var = init_var
        self.theta_hist = []
        self.timestamps = []
        self.basis = basis
        # scale_factors: grid to consider for scaling the prior covariance, (default is that you cannot scale the prior covaraince larger than the given covariance)
        self.covariance_scale_factors = np.exp(np.arange(-3, np.log(max_covariance_scale) + 0.01, 0.05))

    @property
    def num_coef(self):
        return self.init_theta.size

    def get_avg_regret_bound_type_i(self, T: int, embeddings: np.ndarray, n: int):
        """
        @param embeddings: the values of the covariates in the initial recalibration dataset. We use this to estimate R^2 in the theorem
        @return avg regret bound for type I
        """
        embedding_max_eigen = 1/embeddings.shape[0] * np.linalg.eigvalsh(embeddings.T @ embeddings).max()
        regret_bound_type_i = get_regret_bound_type_i(
                self.init_theta,
                self.init_covariance,
                prior_delta2=self.inflation_rate,
                alpha=self.alphas[0],
                T=T,
                n=n,
                embedding_max_eigen=embedding_max_eigen,
            )
        return regret_bound_type_i/T/n

    def _get_avg_regret_bound_type_i_marBLR(self, init_covariance, T: int, embeddings: np.ndarray, n: int, alpha: float):
        """
        @param embeddings: the values of the covariates in the initial recalibration dataset. We use this to estimate R^2 in the theorem
        @return avg regret bound for type I
        """
        embeddings = np.hstack([np.ones((embeddings.shape[0], 1)), embeddings])
        embedding_max_eigen = 1/embeddings.shape[0] * np.linalg.eigvalsh(embeddings.T @ embeddings).max()
        regret_bound_type_i = get_regret_bound_type_i(
                self.init_theta,
                init_covariance,
                prior_delta2=self.inflation_rate,
                alpha=alpha,
                T=T,
                n=n,
                embedding_max_eigen=embedding_max_eigen,
            )
        return regret_bound_type_i/n/T

    def init(self, x, y, tot_time: int, max_regret:float=None, init_theta: np.ndarray = None, init_weights=[1, 0], n=1, ):
        """
        Initialize the recalibrator
        """
        self.x = x
        self.y = y
        self.timepoints = [x.shape[0]]

        # Fit init logistic regression model. Set as prior fixed for now?
        aug_x = self.make_aug_x(x)
        if init_theta is None:
            self.logistic_reg.fit(aug_x, y.ravel())
            self.init_theta = np.concatenate([self.logistic_reg.intercept_, self.logistic_reg.coef_.ravel()])
            init_covariance = get_logistic_cov(aug_x, y, self.init_theta.reshape((-1,1)))
        else:
            # Create a prior matrix
            self.init_theta = init_theta
        init_covariance = get_logistic_cov(aug_x, y, self.init_theta.reshape((-1,1)))

        print("init covairance", init_covariance)
        logging.info("%s, init covariance %s", self.name, init_covariance)
        d_inv = np.diag(1/np.sqrt(np.diag(init_covariance)))
        init_corr = d_inv @ init_covariance @ d_inv
        logging.info("%s, init correlation %s", self.name, init_corr)
        logging.info("%s, init covariance trace %f", self.name, np.trace(init_covariance))
        logging.info("%s, init theta %s", self.name, self.init_theta)
        if max_regret is not None:
            logging.info("tot time %d", tot_time)
            regrets = [
                    self._get_avg_regret_bound_type_i_marBLR(rescale_covariance(init_covariance, scale_factor=factor), tot_time, aug_x, n=n, alpha=self.alphas[0])
                    for factor in self.covariance_scale_factors]
            good_idxs = np.where(regrets < max_regret)[0]
            if good_idxs.size == 0:
                best_factor = self.covariance_scale_factors[0]
                logging.info("COULD NOT SATIFY REGRET")
                logging.info("max regret %f", max_regret)
                logging.info("regrets %s", regrets)
                raise ValueError("TYPE I REGRET NOT GOOD")
            else:
                best_factor = self.covariance_scale_factors[np.max(good_idxs)]
                logging.info("max regret %f", max_regret)
                logging.info("regrets %s", regrets)
                logging.info("best (mar)BLR scale factor %f", best_factor)
                print("best (mar)BLR scale factor %f", best_factor)

            self.init_covariance = rescale_covariance(init_covariance, scale_factor=best_factor)
        else:
            self.init_covariance = init_covariance
        logging.info("%s, scaled covariance trace %f", self.name, np.trace(init_covariance))
        print(self.init_covariance)

        init_posterior = LogisticPosterior(
                theta=self.init_theta,
                covariance=self.init_covariance)

        self.seq_posteriors = [
                ChangepointSeqPosterior([0], init_posterior),
                ChangepointSeqPosterior([1], init_posterior)]
        self.seq_weights = np.array(init_weights).reshape((2,1))
        self.theta_hist.append(self.init_theta.ravel())

    def update(self, x, y, known_refit: bool = False, timestamp = None):
        self.timepoints = self.timepoints + [self.timepoints[-1] + x.shape[0]]


        new_logistic_posteriors = [[None,None] for i in range(2)]
        post_change_tuple = np.zeros((2,2))
        for do_change in self.change_options:
            for did_change, posterior in enumerate(self.seq_posteriors):
                transition_prob = self.alphas[did_change] if (do_change != did_change) else (1 - self.alphas[did_change])
                new_posterior = posterior.extend_seq(do_change, self.inflation_rate, ref_cov=self.init_covariance, max_inflat_rate=self.max_inflat_rate)
                new_logistic_posteriors[do_change][did_change] = new_posterior.logistic_post
                log_score_raw = new_posterior.update(self.make_aug_x(x), y)
                post_change_tuple[do_change, did_change] = np.exp(log_score_raw) * transition_prob * self.seq_weights[did_change]
        post_change_tuple = post_change_tuple/np.sum(post_change_tuple)
        self.seq_weights = post_change_tuple.sum(axis=1)
        # take a mean mixture/collapse
        logistic_posteriors = [
                new_logistic_posteriors[do_change][0].approx_mixture(
                    new_logistic_posteriors[do_change][1],
                    weight=post_change_tuple[do_change,0]/post_change_tuple[do_change,:].sum())
            for do_change in self.change_options
        ]
        self.seq_posteriors = [
            ChangepointSeqPosterior([i], logistic_posteriors[i]) for i in range(2)
        ]

        self.theta_hist.append(self.get_mean_theta().ravel())
        self.timestamps.append(timestamp)

    def predict(self, x):
        aug_x = self.make_aug_x(x)
        new_logistic_posteriors = [[None,None] for i in range(2)]
        final_prediction = 0
        for do_change in self.change_options:
            for did_change, posterior in enumerate(self.seq_posteriors):
                transition_prob = self.alphas[did_change] if (do_change != did_change) else (1 - self.alphas[did_change])
                prediction_case = posterior.extend_seq(do_change, self.inflation_rate).predict(aug_x)
                final_prediction += prediction_case * transition_prob * self.seq_weights[did_change]

        self.print_summary()
        return final_prediction


    def print_summary(self):
        logging.info("%s posterior theta %s", self.name, self.get_mean_theta().ravel())

    @property
    def name(self):
        if self.inflation_rate > 0:
            return "marBLR_%.3f_%.2f_%.4f" % (self.alphas[0], self.alphas[1], self.inflation_rate)
        else:
            return "BLR"

class LockedRecalibrator(Recalibrator):
    is_locked = True
    def init(self, x, y):
        # Fit init logistic regression model. Set as prior fixed for now?
        aug_x = self.make_aug_x(x)
        self.logistic_reg.fit(aug_x, y.ravel())
        self.init_theta = np.concatenate([self.logistic_reg.intercept_, self.logistic_reg.coef_.ravel()])
        init_posterior = LogisticPosterior(
                theta=self.init_theta,
                covariance=None)

        self.seq_posteriors = [ChangepointSeqPosterior([0], init_posterior)]
        self.seq_weights = np.array([1])
        print("locked init theta", self.init_theta)
        logging.info("locked theta %s", self.init_theta)

    def init_score(self, x, y, weight: float=1):
        aug_x = self.make_aug_x(x)
        init_score = get_logistic_score(aug_x, y, self.init_theta, weights=weight)
        return init_score

    def predict(self, x: np.ndarray):
        aug_x = self.make_aug_x(x)
        return self.seq_posteriors[0].logistic_post.predict_map(aug_x)

    def update(self, x, y, known_refit, timestamp=None):
        # do nothing
        return

    def print_summary(self):
        # Do nothing
        return

class AdamRecalibrator(Recalibrator):
    def __init__(self, logistic_reg, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, basis: int = 1):
        self.logistic_reg = logistic_reg
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        self.t = 1
        self.theta_hist = []
        self.timestamps = []
        self.basis = basis

    def init(self, x, y, init_theta: np.ndarray = None):
        # Fit init logistic regression model. Set as prior fixed for now?
        aug_x = self.make_aug_x(x)
        if init_theta is None:
            self.logistic_reg.fit(aug_x, y.ravel())
            self.init_theta = np.concatenate([self.logistic_reg.intercept_, self.logistic_reg.coef_.ravel()])
        else:
            self.init_theta = init_theta
        self.theta = self.init_theta

        self.logistic_mdl = LogisticPosterior(
                theta=self.theta,
                covariance=np.eye(self.theta.size) * 1e-10)

        self.mean_grad = np.zeros((self.theta.size, 1))
        self.var_grad = np.ones((self.theta.size, 1)) * 0.001
        self.seq_posteriors = [
            ChangepointSeqPosterior([0], self.logistic_mdl)
        ]
        self.seq_weights = np.array([1])
        self.theta_hist.append(np.copy(self.theta.ravel()))

    def predict(self, x: np.ndarray):
        aug_x = self.make_aug_x(x)
        return self.seq_posteriors[0].logistic_post.predict_map(aug_x)

    def update(self, x, y, known_refit, timestamp=None):
        aug_x = self.make_aug_x(x)
        y_hat = self.logistic_mdl.predict_map(aug_x)

        aug_x_intercept = np.hstack([np.ones((x.shape[0], 1)), aug_x])
        gradient = -np.mean(((y - y_hat).T @ aug_x_intercept).T, axis=1, keepdims=True)

        self.mean_grad = self.beta1 * self.mean_grad + (1-self.beta1)*gradient

        self.var_grad = self.beta2 * self.var_grad + (1-self.beta2)*(gradient**2)

        ## bias correction
        m_corr = self.mean_grad/(1-self.beta1**self.t)
        v_corr = self.var_grad/(1-self.beta2**self.t)

        ## update weights and biases
        self.logistic_mdl.theta = self.logistic_mdl.theta - self.eta*(m_corr/(np.sqrt(v_corr)+self.epsilon))
        self.t += 1
        self.theta_hist.append(np.copy(self.logistic_mdl.theta.ravel()))
        self.timestamps.append(timestamp)

    def print_summary(self):
        # Do nothing
        return

class CumulativeRecalibrator(Recalibrator):
    eps = 1e-10

    def __init__(self, logistic_reg, basis:int = 1):
        self.logistic_reg = logistic_reg
        self.basis = basis

    def init(self, x, y, init_theta: np.ndarray = None):
        self.x = x
        self.y = y
        # Fit init logistic regression model
        aug_x = self.make_aug_x(x)
        if init_theta is None:
            self.logistic_reg.fit(aug_x, y.ravel())
            self.init_theta = np.concatenate([self.logistic_reg.intercept_, self.logistic_reg.coef_.ravel()])
        else:
            self.init_theta = init_theta

        covariance = np.eye(1 + aug_x.shape[1]) * self.eps
        posterior = LogisticPosterior(
                theta=self.init_theta,
                covariance=covariance)

        self.seq_posteriors = [ChangepointSeqPosterior([0], posterior)]
        self.seq_weights = np.array([1])

    def predict(self, x: np.ndarray):
        aug_x = self.make_aug_x(x)
        return self.seq_posteriors[0].logistic_post.predict_map(aug_x)


    def update(self, x, y, known_refit, timestamp=None):
        self.x = np.vstack([self.x, x])
        self.y = np.vstack([self.y, y])
        # Update logistic regression model
        aug_x = self.make_aug_x(self.x)
        self.logistic_reg.fit(aug_x, self.y.ravel())
        covariance = np.eye(1 + aug_x.shape[1]) * self.eps
        posterior = LogisticPosterior(
                theta=np.concatenate([self.logistic_reg.intercept_, self.logistic_reg.coef_.ravel()]),
                covariance=covariance)

        self.seq_posteriors = [ChangepointSeqPosterior([0], posterior)]

    def print_summary(self):
        # Do nothing
        return

class OracleRecalibrator(Recalibrator):
    def __init__(self, basis: int = 1):
        self.basis = basis

    def init(self, x, y):
        self.theta_hist = []
        self.seq_weights = np.array([1])
        self.do_oracle_update(x, y)

    def do_oracle_update(self, x, y):
        """
        Fit an oracle model.
        If true probabilities are given, fit a logistic regression with true probabilities.
        Otherwise just run logistic regression

        @param y: true probability
        """
        self.x = x
        self.y = y
        aug_x = self.make_aug_x(self.x)

        if np.unique(y).size == 2:
            # Outcomes are binary
            self.logistic_reg = LogisticRegression(penalty="none", solver="lbfgs", warm_start=True)
            self.logistic_reg.fit(aug_x, y.ravel())
            new_theta = np.concatenate([self.logistic_reg.intercept_, self.logistic_reg.coef_.ravel()])
        else:
            # We have the true probabilities
            # Run our custom logistic regression solver
            aug_x = np.hstack([np.ones((x.shape[0], 1)), aug_x])
            new_theta = solve_logistic(aug_x, y)

        posterior = LogisticPosterior(theta=new_theta.ravel(), covariance=None)

        self.seq_posteriors = [ChangepointSeqPosterior([0], posterior)]
        self.theta_hist.append(posterior.theta)

    def predict(self, x: np.ndarray):
        aug_x = self.make_aug_x(x)
        return self.seq_posteriors[0].logistic_post.predict_map(aug_x)

    def update(self, x, y, known_refit, timestamp=None):
        # do nothing
        return

    def print_summary(self):
        # Do nothing
        return
