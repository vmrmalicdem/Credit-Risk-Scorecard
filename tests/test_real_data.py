"""Checks on the real pipeline outputs. Skipped automatically when the data has not been built (for example in CI)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from creditrisk.config import feature_lists
from creditrisk.leakage import check_model_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_TABLE = ROOT / "data" / "processed" / "model_table.parquet"
SPEC = ROOT / "data" / "processed" / "feature_spec.json"
SPLIT = ROOT / "artifacts" / "full" / "split.parquet"


@pytest.mark.skipif(not SPEC.exists(), reason="run build_dataset first")
def test_saved_feature_spec_matches_the_config_and_has_no_outcome_fields(cfg, schema):
    spec = json.loads(SPEC.read_text())
    assert spec == feature_lists(cfg, "full")
    check_model_features(spec["numeric"] + spec["categorical"], schema)


@pytest.mark.skipif(not MODEL_TABLE.exists(), reason="run build_dataset first")
def test_model_table_has_one_row_per_loan_and_binary_target():
    df = pd.read_parquet(MODEL_TABLE)
    assert df.loan_id.is_unique
    assert set(df.target.unique()) <= {0, 1}
    assert df.target.sum() > 0
    assert (df.ead > 0).all()


@pytest.mark.skipif(not (MODEL_TABLE.exists() and SPLIT.exists()), reason="run build_dataset and train first")
def test_split_is_loan_level_complete_and_stratified():
    df = pd.read_parquet(MODEL_TABLE)
    s = pd.read_parquet(SPLIT)
    assert s.loan_id.is_unique and set(s.loan_id) == set(df.loan_id)
    merged = df[["loan_id", "target"]].merge(s, on="loan_id")
    rate = merged.groupby("split").target.mean()
    assert abs(rate["dev"] - rate["test"]) < 0.25 * merged.target.mean()
