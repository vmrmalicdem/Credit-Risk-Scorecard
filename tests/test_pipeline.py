"""Pipeline tests: unit tests always run; data tests are skipped until the pipeline has produced its files."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from features import QuantileBinner

ROOT = Path(__file__).resolve().parents[1]
MODEL_TABLE = ROOT / "data" / "processed" / "model_table.parquet"
SPEC = ROOT / "data" / "processed" / "feature_spec.json"
SPLIT = ROOT / "artifacts" / "full" / "split.parquet"


def test_binner_separates_most_common_value_from_larger_values():
    x = pd.DataFrame({"a": [0] * 95 + [1] * 4 + [2]})
    b = QuantileBinner(5).fit(x)
    counts = b.transform(x).sum(axis=0)
    assert list(b.get_feature_names_out()) == ["a__bin0", "a__bin1"]
    assert list(counts) == [95, 5]


def test_binner_gives_missing_values_their_own_column():
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan]})
    b = QuantileBinner(2).fit(x)
    m = b.transform(x)
    assert b.get_feature_names_out()[-1] == "a__missing"
    assert m[-1, -1] == 1 and m[:-1, -1].sum() == 0
    assert (m.sum(axis=1) == 1).all()


@pytest.mark.skipif(not SPEC.exists(), reason="run build_dataset.py first")
def test_no_outcome_or_meta_fields_are_features():
    schema = yaml.safe_load(open(ROOT / "configs" / "schema_map.yaml"))
    outcome = {c["name"] for c in schema["origination"] + schema["performance"] if c["known_at"] == "outcome"}
    spec = json.load(open(SPEC))
    features = set(spec["numeric"] + spec["categorical"])
    assert not features & outcome
    assert not any(f.startswith("meta_") or f in {"target", "ead", "loan_id"} for f in features)


@pytest.mark.skipif(not MODEL_TABLE.exists(), reason="run build_dataset.py first")
def test_model_table_has_one_row_per_loan_and_binary_target():
    df = pd.read_parquet(MODEL_TABLE)
    assert df.loan_id.is_unique
    assert set(df.target.unique()) <= {0, 1}
    assert df.target.sum() > 0
    assert (df.ead > 0).all()


@pytest.mark.skipif(not (MODEL_TABLE.exists() and SPLIT.exists()), reason="run build_dataset.py and train.py first")
def test_split_is_loan_level_complete_and_stratified():
    df = pd.read_parquet(MODEL_TABLE)
    s = pd.read_parquet(SPLIT)
    assert s.loan_id.is_unique and set(s.loan_id) == set(df.loan_id)
    merged = df[["loan_id", "target"]].merge(s, on="loan_id")
    rate = merged.groupby("split").target.mean()
    assert abs(rate["dev"] - rate["test"]) < 0.25 * merged.target.mean()
