import copy

import numpy as np
import pandas as pd
import pytest
from synth import make_valid_table

from creditrisk.validation import (
    DataValidationError,
    validate_model_table,
    validate_origination,
    validate_scoring_input,
)


@pytest.fixture(scope="module")
def table(cfg):
    return make_valid_table(cfg)


@pytest.fixture
def strict_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["validation"]["tables"]["model_table"].update(min_rows=1000, min_defaults=1, default_rate=[0.0001, 0.5])
    return c


def fails_with(fn, *args, text):
    with pytest.raises(DataValidationError) as e:
        fn(*args)
    assert text in str(e.value)


def test_valid_table_passes_and_profile_covers_every_column(strict_cfg, table):
    prof, failures = validate_model_table(table, strict_cfg)
    assert failures.empty
    assert prof.column.is_unique and {"credit_score", "target", "loan_id"} <= set(prof.column)
    assert (prof.null_rate <= prof.max_null_rate).all() or prof.max_null_rate.eq(1).any()


@pytest.mark.parametrize("column, value, text", [
    ("credit_score", 120, "greater_than_or_equal_to(300)"),
    ("credit_score", 990, "less_than_or_equal_to(850)"),
    ("ltv", 0, "greater_than_or_equal_to(1)"),
    ("dti", 140, "less_than_or_equal_to(100)"),
    ("orig_upb", 5_000_000, "less_than_or_equal_to(2000000)"),
    ("orig_rate", -1.0, "greater_than_or_equal_to(0.5)"),
    ("occupancy", "X", "isin"),
    ("state", "Texas", "str_matches"),
    ("target", 2, "isin"),
    ("pred_period", "2016-12", "str_matches"),
    ("ead", 0, "greater_than_or_equal_to(1)"),
])
def test_out_of_range_or_disallowed_values_are_rejected(strict_cfg, table, column, value, text):
    bad = table.copy()
    bad[column] = bad[column].astype(object) if isinstance(value, str) else bad[column]
    bad.loc[bad.index[0], column] = value
    fails_with(validate_model_table, bad, strict_cfg, text=text)


def test_null_rate_above_limit_is_rejected_and_below_is_allowed(strict_cfg, table):
    bad = table.copy()
    bad.loc[bad.sample(frac=0.10, random_state=0).index, "credit_score"] = np.nan  # limit is 2%
    fails_with(validate_model_table, bad, strict_cfg, text="null rate of credit_score")
    ok = table.copy()
    ok.loc[ok.index[:5], "credit_score"] = np.nan
    assert validate_model_table(ok, strict_cfg)[1].empty


def test_nulls_in_a_column_with_no_allowance_are_rejected(strict_cfg, table):
    bad = table.copy()
    bad.loc[bad.index[0], "orig_upb"] = np.nan
    with pytest.raises(DataValidationError):
        validate_model_table(bad, strict_cfg)


def test_duplicate_loans_and_missing_columns_are_rejected(strict_cfg, table):
    fails_with(validate_model_table, pd.concat([table, table.iloc[:3]]), strict_cfg, text="loan_id")
    fails_with(validate_model_table, table.drop(columns=["dti"]), strict_cfg, text="missing columns")


def test_table_level_rules(strict_cfg, table):
    fails_with(validate_model_table, table.iloc[:200], strict_cfg, text="too few rows")
    none = table.assign(target=0)
    fails_with(validate_model_table, none, strict_cfg, text="too few defaults")
    many = table.assign(target=1)
    fails_with(validate_model_table, many, strict_cfg, text="default rate")


def test_every_failure_is_listed_not_just_the_first(strict_cfg, table):
    bad = table.copy()
    bad.loc[bad.index[0], "credit_score"] = 120
    bad.loc[bad.index[1], "dti"] = 140
    with pytest.raises(DataValidationError) as e:
        validate_model_table(bad, strict_cfg)
    assert "credit_score" in str(e.value) and "dti" in str(e.value)


def test_warn_mode_returns_the_failures_instead_of_raising(strict_cfg, table):
    strict_cfg["validation"]["on_failure"] = "warn"
    bad = table.copy()
    bad.loc[bad.index[0], "credit_score"] = 120
    with pytest.warns(UserWarning, match="credit_score"):
        _, failures = validate_model_table(bad, strict_cfg)
    assert len(failures) >= 1


def test_every_configured_feature_column_has_a_rule(cfg):
    from creditrisk.config import feature_lists

    spec = feature_lists(cfg, "full")
    assert set(spec["numeric"] + spec["categorical"]) <= set(cfg["validation"]["columns"])


def test_scoring_input_checks_values_but_not_batch_null_rates(strict_cfg, table):
    spec = {"numeric": ["credit_score", "ltv"], "categorical": ["occupancy"]}
    batch = table[["credit_score", "ltv", "occupancy"]].head(20).copy()
    batch.loc[batch.index[:10], "credit_score"] = np.nan  # a small batch can have many missing values
    assert validate_scoring_input(batch, strict_cfg, spec)[1].empty
    batch.loc[batch.index[11], "ltv"] = 5000
    fails_with(validate_scoring_input, batch, strict_cfg, spec, text="less_than_or_equal_to(998)")
    fails_with(validate_scoring_input, batch.drop(columns=["ltv"]), strict_cfg, spec, text="missing columns")


def test_origination_validation_uses_only_origination_columns(strict_cfg, table):
    orig = table.drop(columns=["upb_ratio", "delinq_at_pred", "max_delinq_prior12", "ever_90_before", "mod_at_pred"])
    assert validate_origination(orig, strict_cfg)[1].empty
