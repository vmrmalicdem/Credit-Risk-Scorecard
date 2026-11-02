import numpy as np
import pandas as pd

from creditrisk.features import QuantileBinner


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


def test_every_row_falls_in_exactly_one_bin():
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.normal(size=500), "b": rng.integers(0, 4, 500)})
    m = QuantileBinner(5).fit(x).transform(x)
    assert (m.sum(axis=1) == 2).all()


def test_bins_are_roughly_equal_frequency_for_continuous_data():
    x = pd.DataFrame({"a": np.arange(1000, dtype=float)})
    counts = QuantileBinner(5).fit(x).transform(x).sum(axis=0)
    assert counts.min() >= 195 and counts.max() <= 205


def test_values_beyond_the_training_range_go_to_the_end_bins():
    b = QuantileBinner(4).fit(pd.DataFrame({"a": np.arange(100, dtype=float)}))
    m = b.transform(pd.DataFrame({"a": [-1e9, 1e9]}))
    assert m[0, 0] == 1 and m[1, -1] == 1


def test_value_equal_to_an_edge_goes_to_the_lower_bin():
    b = QuantileBinner(2).fit(pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}))
    edge = b.edges_["a"][0]
    m = b.transform(pd.DataFrame({"a": [edge]}))
    assert m[0, 0] == 1


def test_missing_value_unseen_in_training_gets_an_all_zero_row():
    b = QuantileBinner(2).fit(pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}))
    m = b.transform(pd.DataFrame({"a": [np.nan]}))
    assert m.sum() == 0


def test_binning_is_deterministic_and_does_not_change_with_row_order():
    rng = np.random.default_rng(1)
    x = pd.DataFrame({"a": rng.normal(size=300)})
    shuffled = x.sample(frac=1, random_state=0)
    e1 = QuantileBinner(5).fit(x).edges_["a"]
    e2 = QuantileBinner(5).fit(shuffled).edges_["a"]
    assert np.array_equal(e1, e2)
