"""
Helper functions with logistic regression
Mostly to get the standard error matrix
"""

import numpy as np

def get_logistic_score(x, y, theta, weights: float = 1):
    intercept_x = np.hstack([np.ones((x.shape[0], 1)), x])
    logit_preds = np.array(np.matmul(intercept_x, theta.reshape((-1,1))), dtype=float)
    p_hat = (1/(1 + np.exp(-logit_preds))).flatten()
    y_flat = y.ravel()
    log_liks = (np.log(p_hat) * y_flat + np.log(1 - p_hat) * (1 - y_flat)) * weights.ravel()
    return -np.sum(log_liks)/np.sum(weights)

def get_logistic_cov(x, y, theta):
    """
    @return inverse hessian, empirical fishers matrix
    """
    intercept_x = np.hstack([np.ones((x.shape[0], 1)), x])

    logit_preds = np.array(np.matmul(intercept_x, theta), dtype=float)
    p_hat = (1/(1 + np.exp(-logit_preds))).flatten()
    variance_hat_mat = np.diag(p_hat * (1 - p_hat))
    hessian = intercept_x.T @ variance_hat_mat @ intercept_x
    cov_est = np.linalg.pinv(hessian)
    return cov_est

def rescale_covariance(cov_mat: np.ndarray, scale_factor: float = None, use_diag: bool = False):
    if use_diag:
        corr_mat = np.eye(cov_mat.shape[0])
    else:
        d_inv = np.diag(1/np.sqrt(np.diag(cov_mat)))
        corr_mat = d_inv @ cov_mat @ d_inv

    if scale_factor is not None:
        return scale_factor * cov_mat
    else:
        return cov_mat

