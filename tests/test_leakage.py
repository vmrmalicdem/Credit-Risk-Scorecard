"""Leakage tests: no post-origination or outcome field may enter the feature set."""
import pandas as pd
import pytest
from synth import PRED_IDX, write_raw

from creditrisk.build_dataset import build
from creditrisk.config import feature_lists
from creditrisk.leakage import LeakageError, check_feature_config, check_model_features, schema_tags


def test_shipped_config_passes(cfg, schema):
    check_feature_config(cfg, schema)


@pytest.mark.parametrize("outcome_field", ["actual_loss", "zero_bal_code", "zero_bal_removal_upb", "mi_recoveries"])
def test_outcome_field_as_origination_feature_fails(cfg_copy, schema, outcome_field):
    cfg_copy["features"]["origination"]["numeric"].append(outcome_field)
    with pytest.raises(LeakageError):
        check_feature_config(cfg_copy, schema)


@pytest.mark.parametrize("performance_field", ["current_upb", "delinq_status", "current_rate"])
def test_performance_field_as_origination_feature_fails(cfg_copy, schema, performance_field):
    cfg_copy["features"]["origination"]["numeric"].append(performance_field)
    with pytest.raises(LeakageError):
        check_feature_config(cfg_copy, schema)


@pytest.mark.parametrize("outcome_field", ["zero_bal_code", "actual_loss", "net_sale_proceeds"])
def test_behaviour_feature_sourced_from_outcome_field_fails(cfg_copy, schema, outcome_field):
    cfg_copy["features"]["behaviour"]["numeric"]["upb_ratio"] = ["current_upb", outcome_field]
    with pytest.raises(LeakageError):
        check_feature_config(cfg_copy, schema)


def test_behaviour_feature_without_declared_source_fails(cfg_copy, schema):
    cfg_copy["features"]["behaviour"]["numeric"]["new_feature"] = []
    with pytest.raises(LeakageError):
        check_feature_config(cfg_copy, schema)


@pytest.mark.parametrize(
    "name", ["target", "ead", "loan_id", "pred_period", "meta_disaster_event", "meta_assist_event"]
)
def test_reserved_and_reporting_columns_fail(cfg_copy, schema, name):
    cfg_copy["features"]["origination"]["numeric"].append(name)
    with pytest.raises(LeakageError):
        check_feature_config(cfg_copy, schema)


def test_duplicate_feature_names_fail(cfg_copy, schema):
    cfg_copy["features"]["origination"]["categorical"].append("state")
    with pytest.raises(LeakageError):
        check_feature_config(cfg_copy, schema)


def test_model_feature_check_rejects_outcome_and_meta_columns(schema):
    with pytest.raises(LeakageError):
        check_model_features(["credit_score", "actual_loss"], schema)
    with pytest.raises(LeakageError):
        check_model_features(["credit_score", "meta_assist_event"], schema)
    check_model_features(["credit_score", "ltv"], schema)


@pytest.mark.parametrize("feature_set", ["full", "origination"])
def test_every_modeled_feature_is_allowed(cfg, schema, feature_set):
    spec = feature_lists(cfg, feature_set)
    check_model_features(spec["numeric"] + spec["categorical"], schema)


def test_origination_set_contains_only_origination_tagged_fields(cfg, schema):
    tags = schema_tags(schema)
    spec = feature_lists(cfg, "origination")
    assert {tags[f] for f in spec["numeric"] + spec["categorical"]} == {"origination"}


def test_built_table_features_exclude_outcome_columns(synthetic, cfg, schema):
    tags = schema_tags(schema)
    spec = feature_lists(cfg, "full")
    feature_cols = spec["numeric"] + spec["categorical"]
    assert set(feature_cols) <= set(synthetic["model"].columns)
    assert not any(tags.get(c) == "outcome" for c in feature_cols)


def test_features_do_not_change_when_data_after_the_prediction_month_changes(synthetic, cfg, schema, tmp_path):
    """The decisive leakage test: rewrite everything that happens after the prediction month and re-build.
    Features must be identical; only the target may move."""
    perf = synthetic["perf"].copy()
    period_idx = perf.period.astype(int).map(lambda v: (v // 100) * 12 + v % 100 - 1)
    after = period_idx > PRED_IDX
    perf.loc[after, "delinq_status"] = "05"
    perf.loc[after, "borrower_assist_plan"] = "F"
    perf.loc[after, "disaster_delinq_flag"] = "Y"
    perf.loc[after, "actual_loss"] = "99999.00"
    perf.loc[after, "zero_bal_removal_upb"] = "1.00"
    o_path, p_path = write_raw(tmp_path, schema, synthetic["orig"], perf)
    changed, _, _ = build(cfg, schema, o_path, p_path)

    spec = feature_lists(cfg, "full")
    cols = ["loan_id"] + spec["numeric"] + spec["categorical"] + ["ead", "pred_period"]
    base = synthetic["model"][cols].set_index("loan_id")
    other = changed[cols].set_index("loan_id").reindex(base.index)
    pd.testing.assert_frame_equal(base, other)
    assert changed.set_index("loan_id").loc[base.index, "target"].eq(1).all()
    assert synthetic["model"].target.sum() < len(synthetic["model"])


def test_features_do_not_change_when_only_outcome_columns_change(synthetic, cfg, schema, tmp_path):
    perf = synthetic["perf"].copy()
    perf["actual_loss"] = "12345.00"
    perf["zero_bal_removal_upb"] = "50000.00"
    o_path, p_path = write_raw(tmp_path, schema, synthetic["orig"], perf)
    changed, _, _ = build(cfg, schema, o_path, p_path)
    spec = feature_lists(cfg, "full")
    cols = ["loan_id"] + spec["numeric"] + spec["categorical"] + ["target"]
    pd.testing.assert_frame_equal(synthetic["model"][cols], changed[cols])
