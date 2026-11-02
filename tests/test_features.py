import numpy as np
import pandas as pd
import pytest

from creditrisk.config import feature_lists
from creditrisk.features import category_levels, lgbm_frame, scorecard_frame


def test_full_set_is_origination_plus_behaviour(cfg):
    full, orig = feature_lists(cfg, "full"), feature_lists(cfg, "origination")
    assert full["numeric"][: len(orig["numeric"])] == orig["numeric"]
    assert full["categorical"][: len(orig["categorical"])] == orig["categorical"]
    behaviour = set(cfg["features"]["behaviour"]["numeric"]) | set(cfg["features"]["behaviour"]["categorical"])
    assert set(full["numeric"] + full["categorical"]) - set(orig["numeric"] + orig["categorical"]) == behaviour


def test_unknown_feature_set_is_rejected(cfg):
    with pytest.raises(ValueError):
        feature_lists(cfg, "everything")


def test_feature_lists_have_no_duplicates(cfg):
    for fs in ("full", "origination"):
        spec = feature_lists(cfg, fs)
        names = spec["numeric"] + spec["categorical"]
        assert len(names) == len(set(names))


def test_scorecard_frame_fills_missing_categories_and_keeps_numeric_nans():
    df = pd.DataFrame({"n": [1.0, np.nan], "c": ["A", None], "extra": [9, 9]})
    out = scorecard_frame(df, {"numeric": ["n"], "categorical": ["c"]})
    assert list(out.columns) == ["n", "c"]
    assert out.c.tolist() == ["A", "MISSING"]
    assert out.n.isna().tolist() == [False, True]


def test_lgbm_frame_uses_fixed_levels_so_unseen_values_become_missing():
    df = pd.DataFrame({"n": [1.0, 2.0], "c": ["A", "Z"]})
    out = lgbm_frame(df, {"numeric": ["n"], "categorical": ["c"]}, {"c": ["A", "B"]})
    assert list(out.c.cat.categories) == ["A", "B"]
    assert out.c.isna().tolist() == [False, True]


def test_category_levels_are_sorted_and_drop_missing():
    df = pd.DataFrame({"c": ["b", "a", None, "b"]})
    assert category_levels(df, {"categorical": ["c"]}) == {"c": ["a", "b"]}
