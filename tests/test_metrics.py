import numpy as np
import pytest

from creditrisk.metrics import (
    apply_platt,
    bootstrap_indices,
    bootstrap_table,
    calibration_bins,
    clip_prob,
    fit_platt,
    point_metrics,
)

EPS, C = 1e-6, 1e6


def test_clip_keeps_probabilities_strictly_inside_zero_one():
    out = clip_prob(np.array([0.0, 0.5, 1.0]), EPS)
    assert out[0] == EPS and out[2] == 1 - EPS and out[1] == 0.5


def test_platt_is_near_identity_for_well_calibrated_predictions():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.3, 40_000)
    y = (rng.random(40_000) < p).astype(int)
    a, b = fit_platt(p, y, EPS, C)
    assert abs(a) < 0.1 and abs(b - 1) < 0.1


def test_platt_corrects_overconfident_predictions():
    rng = np.random.default_rng(1)
    true = rng.uniform(0.02, 0.2, 40_000)
    y = (rng.random(40_000) < true).astype(int)
    logit = np.log(true / (1 - true))
    overconfident = 1 / (1 + np.exp(-2 * logit))
    ab = fit_platt(overconfident, y, EPS, C)
    assert ab[1] == pytest.approx(0.5, abs=0.1)
    assert abs(apply_platt(overconfident, ab, EPS).mean() - y.mean()) < 0.01


def test_point_metrics_on_a_perfect_ranking():
    y = np.array([0] * 90 + [1] * 10)
    p = np.concatenate([np.linspace(0.001, 0.05, 90), np.linspace(0.5, 0.9, 10)])
    m = point_metrics(y, p, threshold=0.4, eps=EPS, c=C)
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == pytest.approx(1.0)
    assert m["recall_top10"] == 1.0 and m["precision_top10"] == 1.0
    assert m["base_rate"] == 0.1


def test_bootstrap_interval_contains_the_estimate_and_is_reproducible():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 800)
    y = (rng.random(800) < p * 0.2).astype(int)
    idx = bootstrap_indices(len(y), 60, np.random.default_rng(3))
    est, rows = bootstrap_table("m", y, p, np.quantile(p, 0.9), idx, EPS, C)
    auc = next(r for r in rows if r["metric"] == "roc_auc")
    assert auc["ci_low"] <= est["roc_auc"] <= auc["ci_high"]
    idx2 = bootstrap_indices(len(y), 60, np.random.default_rng(3))
    assert all((a == b).all() for a, b in zip(idx, idx2, strict=True))


def test_calibration_bins_partition_the_loans():
    rng = np.random.default_rng(4)
    p = rng.uniform(0, 0.1, 1000)
    y = (rng.random(1000) < p).astype(int)
    t = calibration_bins("m", y, p, 5)
    assert t.loans.sum() == 1000 and t.events.sum() == y.sum() and len(t) == 5
    assert t.mean_pred.is_monotonic_increasing
