"""Train/serve skew: scoring must reproduce training-time transformations exactly, from raw loan columns."""
import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from synth import make_valid_table

from creditrisk.config import feature_lists
from creditrisk.features import category_levels
from creditrisk.metrics import fit_platt
from creditrisk.model import CalibratedScorer, build_lgbm_pipeline, build_scorecard_pipeline
from creditrisk.score import score_loans

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data(cfg):
    df = make_valid_table(cfg, n=3000)
    return df.iloc[:2000].reset_index(drop=True), df.iloc[2000:].reset_index(drop=True)


@pytest.fixture(scope="module")
def spec(cfg):
    return feature_lists(cfg, "full")


@pytest.fixture(scope="module", params=["scorecard", "lgbm"])
def scorer(request, cfg, spec, data):
    dev, _ = data
    small = copy.deepcopy(cfg)
    if request.param == "scorecard":
        pipe = build_scorecard_pipeline(spec, small)
    else:
        params = {k: v[0] for k, v in small["lgbm"]["grid"].items()}
        small["lgbm"]["fixed"]["n_estimators"] = 30
        pipe = build_lgbm_pipeline(spec, small, category_levels(dev, spec), params=params)
    y = dev.target.to_numpy()
    pipe.fit(dev, y)
    eps = cfg["evaluation"]["prob_clip"]
    platt = fit_platt(pipe.predict_proba(dev)[:, 1], y, eps, cfg["evaluation"]["platt_c"])
    return CalibratedScorer(pipe, platt, eps)


def test_pipeline_equals_applying_each_step_by_hand(scorer, data):
    _, new = data
    pipe = scorer.pipeline
    X = new
    for _, step in pipe.steps[:-1]:
        X = step.transform(X)
    by_hand = pipe.steps[-1][1].predict_proba(X)[:, 1]
    assert np.array_equal(by_hand, scorer.predict_raw(new))


def test_saved_and_reloaded_scorer_gives_identical_probabilities(scorer, data, tmp_path):
    _, new = data
    joblib.dump(scorer, tmp_path / "m.joblib")
    again = joblib.load(tmp_path / "m.joblib")
    assert np.array_equal(again.predict_proba(new), scorer.predict_proba(new))


def test_column_order_and_extra_columns_do_not_change_scores(scorer, data):
    _, new = data
    shuffled = new[list(reversed(new.columns))].assign(unrelated=1.0, target_leak=new.target)
    assert np.array_equal(scorer.predict_proba(shuffled), scorer.predict_proba(new))


def test_a_loan_scores_the_same_alone_as_inside_a_batch(scorer, data):
    _, new = data
    batch = scorer.predict_proba(new)[:, 1]
    for i in (0, 17, 500):
        assert scorer.predict_proba(new.iloc[[i]])[0, 1] == pytest.approx(batch[i], rel=1e-12)


def test_missing_feature_column_fails_loudly_naming_the_column(scorer, data):
    _, new = data
    with pytest.raises(ValueError, match="ltv"):
        scorer.predict_proba(new.drop(columns=["ltv"]))


def test_unseen_category_is_handled_like_a_missing_value(scorer, data):
    _, new = data
    unseen, missing = new.copy(), new.copy()
    unseen["state"] = "ZZ"
    missing["state"] = None
    p = scorer.predict_proba(unseen)[:, 1]
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
    if "lgbm" in type(scorer.pipeline.steps[-1][1]).__name__.lower():
        assert np.array_equal(p, scorer.predict_proba(missing)[:, 1])


def test_scoring_input_with_an_out_of_range_value_is_rejected_before_the_model(scorer, data, cfg):
    from creditrisk.validation import DataValidationError

    _, new = data
    bad = new.copy()
    bad.loc[0, "credit_score"] = 90
    with pytest.raises(DataValidationError, match="credit_score"):
        score_loans(bad, scorer, cfg)


def test_score_loans_returns_probability_and_points_for_every_loan(scorer, data, cfg):
    _, new = data
    out = score_loans(new, scorer, cfg)
    assert list(out.columns) == ["loan_id", "pd", "score"] and len(out) == len(new)
    assert out.pd.between(0, 1).all()
    assert out.score.corr(out.pd, method="spearman") == pytest.approx(-1.0)


@pytest.mark.parametrize("fs", ["full", "origination"])
@pytest.mark.parametrize("name", ["scorecard", "lgbm"])
def test_saved_models_reproduce_the_saved_test_predictions(fs, name, cfg):
    art, table = ROOT / "artifacts" / fs, ROOT / "data" / "processed" / "model_table.parquet"
    if not ((art / f"{name}.joblib").exists() and table.exists()):
        pytest.skip("run the pipeline first")
    split = pd.read_parquet(art / "split.parquet").set_index("loan_id").split
    pred = pd.read_parquet(art / "predictions.parquet").set_index("loan_id")
    df = pd.read_parquet(table)
    test = df[df.loan_id.map(split) == "test"].reset_index(drop=True)
    p = joblib.load(art / f"{name}.joblib").predict_proba(test)[:, 1]
    assert np.allclose(p, pred.loc[test.loan_id, f"p_{name}_platt"].to_numpy(), atol=1e-12)
