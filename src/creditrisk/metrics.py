"""Probability calibration and test-set metrics with bootstrap intervals."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def clip_prob(p, eps):
    return np.clip(p, eps, 1 - eps)


def fit_platt(raw_oof, y, eps, c):
    """Logistic recalibration on out-of-fold predictions; returns (intercept, slope) on the logit scale."""
    fit = LogisticRegression(C=c).fit(logit(clip_prob(raw_oof, eps)).reshape(-1, 1), y)
    return float(fit.intercept_[0]), float(fit.coef_[0, 0])


def apply_platt(p, ab, eps):
    return expit(ab[0] + ab[1] * logit(clip_prob(p, eps)))


def point_metrics(y, p, threshold, eps, c):
    z = logit(clip_prob(p, eps)).reshape(-1, 1)
    cal = LogisticRegression(C=c).fit(z, y)
    flag = p >= threshold
    return {
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
        "cal_intercept": cal.intercept_[0], "cal_slope": cal.coef_[0, 0],
        "recall_top10": y[flag].sum() / y.sum(), "precision_top10": y[flag].mean() if flag.any() else np.nan,
        "mean_pred": p.mean(), "base_rate": y.mean(),
    }


def bootstrap_indices(n, n_boot, rng):
    return [rng.integers(0, n, n) for _ in range(n_boot)]


def bootstrap_table(variant, y, p, threshold, boot_idx, eps, c):
    """Rows of (model, metric, estimate, ci_low, ci_high) with percentile bootstrap intervals."""
    est = point_metrics(y, p, threshold, eps, c)
    boots = pd.DataFrame([point_metrics(y[i], p[i], threshold, eps, c) for i in boot_idx])
    rows = [{"model": variant, "metric": m, "estimate": v,
             "ci_low": boots[m].quantile(0.025), "ci_high": boots[m].quantile(0.975)} for m, v in est.items()]
    return est, rows


def calibration_bins(variant, y, p, n_bins):
    q = pd.qcut(pd.Series(p).rank(method="first"), n_bins, labels=False)
    t = pd.DataFrame({"q": q, "p": p, "y": y}).groupby("q").agg(
        loans=("y", "size"), events=("y", "sum"), mean_pred=("p", "mean"), obs_rate=("y", "mean")).reset_index()
    t.insert(0, "model", variant)
    return t
