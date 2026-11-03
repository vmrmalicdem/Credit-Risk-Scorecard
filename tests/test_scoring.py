"""Score calculation: points scaling and the exact equivalence between points and the logistic model."""
import numpy as np
import pandas as pd
import pytest

from creditrisk.scoring import (
    coefficients_to_points,
    feature_bins,
    numeric_labels,
    pd_to_score,
    points_range_by_feature,
    points_scale,
    score_from_design,
    scorecard_table,
)


@pytest.fixture(scope="module")
def scale(cfg):
    p = cfg["points"]
    return points_scale(p["base_score"], p["base_odds"], p["pdo"])


def test_base_score_is_reached_at_base_odds(cfg, scale):
    factor, offset = scale
    p = cfg["points"]
    pd_at_base = 1 / (1 + p["base_odds"])
    assert pd_to_score(pd_at_base, factor, offset) == pytest.approx(p["base_score"])


def test_pdo_points_double_the_odds(cfg, scale):
    factor, offset = scale
    p = cfg["points"]
    odds = np.array([50.0, 200.0, 800.0])
    s1 = pd_to_score(1 / (1 + odds), factor, offset)
    s2 = pd_to_score(1 / (1 + 2 * odds), factor, offset)
    assert np.allclose(s2 - s1, p["pdo"])


def test_lower_default_probability_gives_higher_score(scale):
    factor, offset = scale
    s = pd_to_score(np.array([0.001, 0.01, 0.1, 0.5]), factor, offset)
    assert (np.diff(s) < 0).all()


def test_coefficients_to_points_signs_and_base(scale):
    factor, offset = scale
    pts, base = coefficients_to_points([0.5, -0.5, 0.0], intercept=-4.0, factor=factor, offset=offset)
    assert pts[0] < 0 < pts[1] and pts[2] == 0
    assert base == pytest.approx(offset + 4.0 * factor)


def test_points_total_equals_scale_applied_to_the_model_log_odds(fitted_scorecard, model_table, scale):
    pipe, spec = fitted_scorecard
    factor, offset = scale
    X = model_table
    design = pipe[:-1].transform(X)
    pts, base = coefficients_to_points(pipe.named_steps["lr"].coef_[0], pipe.named_steps["lr"].intercept_[0],
                                       factor, offset)
    from_points = score_from_design(design, pts, base)
    from_model = offset - factor * pipe.decision_function(X)
    assert np.allclose(from_points, from_model)
    p = pipe.predict_proba(X)[:, 1]
    assert np.allclose(from_points, pd_to_score(p, factor, offset))


def test_scorecard_table_reproduces_the_model_score(fitted_scorecard, model_table, scale):
    pipe, spec = fitted_scorecard
    factor, offset = scale
    table = scorecard_table(pipe, model_table, scale)
    assert table.feature.iloc[0] == "BASE"
    body = table.iloc[1:]
    design = pipe[:-1].transform(model_table)
    total = table.points.iloc[0] + np.asarray(design @ body.points.to_numpy()).ravel()
    assert np.allclose(total, offset - factor * pipe.decision_function(model_table))


def test_scorecard_table_counts_cover_every_loan_once_per_feature(fitted_scorecard, model_table, scale):
    pipe, spec = fitted_scorecard
    body = scorecard_table(pipe, model_table, scale).iloc[1:]
    per_feature = body.groupby("feature").loans_dev.sum()
    expected = len(model_table)
    # an unseen numeric value would add nothing; every feature here sees every loan in exactly one bin
    assert (per_feature == expected).all()
    assert body.defaults_dev.groupby(body.feature).sum().eq(model_table.target.sum()).all()


def test_default_rate_per_bin_is_defaults_over_loans(fitted_scorecard, model_table, scale):
    pipe, spec = fitted_scorecard
    body = scorecard_table(pipe, model_table, scale).iloc[1:]
    ok = body.loans_dev > 0
    assert np.allclose(body.default_rate_dev[ok], body.defaults_dev[ok] / body.loans_dev[ok])


def test_bin_labels_match_design_matrix_width(fitted_scorecard):
    pipe, spec = fitted_scorecard
    pre = pipe.named_steps["pre"]
    labels = numeric_labels(pre.named_transformers_["num"])
    n_num = sum(len(v) for v in labels.values())
    assert n_num == len(pre.named_transformers_["num"].get_feature_names_out())
    assert len(feature_bins(pre, spec)) == len(pre.get_feature_names_out())


def test_numeric_bin_labels_are_ordered_edges(fitted_scorecard):
    pipe, _ = fitted_scorecard
    labels = numeric_labels(pipe.named_steps["pre"].named_transformers_["num"])["credit_score"]
    assert labels[0].startswith("<=") and labels[-1].startswith(">")


def test_higher_risk_inputs_score_lower_on_the_fitted_scorecard(fitted_scorecard, model_table, scale):
    pipe, spec = fitted_scorecard
    factor, offset = scale
    base = model_table.iloc[[0]].copy()
    safe, risky = base.copy(), base.copy()
    safe["ltv"], risky["ltv"] = 40.0, 95.0
    safe["delinq_at_pred"], risky["delinq_at_pred"] = 0.0, 2.0
    both = pd.concat([safe, risky], ignore_index=True)
    s = offset - factor * pipe.decision_function(both)
    assert s[0] > s[1]


def test_points_range_ignores_sparse_bins():
    t = pd.DataFrame({"feature": ["BASE", "a", "a", "a", "b", "b"],
                      "bin": ["i", "x", "y", "z", "x", "y"],
                      "points": [500.0, 10.0, -5.0, 90.0, 4.0, -4.0],
                      "loans_dev": [100, 80, 60, 3, 50, 50]})
    r = points_range_by_feature(t, min_loans=50)
    assert r["a"] == 15.0 and r["b"] == 8.0
    assert list(r.index) == ["a", "b"]
