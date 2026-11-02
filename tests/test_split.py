import numpy as np

from creditrisk.train import make_split


def test_split_is_disjoint_complete_and_stratified(cfg, model_table):
    dev, test = make_split(model_table, cfg)
    assert set(dev).isdisjoint(test) and len(dev) + len(test) == len(model_table)
    assert abs(len(test) / len(model_table) - cfg["split"]["test_size"]) < 0.01
    y = model_table.target.to_numpy()
    assert abs(y[dev].mean() - y[test].mean()) < 0.1 * y.mean() + 1e-3
    assert y[dev].sum() + y[test].sum() == y.sum()


def test_split_is_reproducible_and_depends_on_the_seed(cfg, model_table):
    a, _ = make_split(model_table, cfg)
    b, _ = make_split(model_table, cfg)
    assert np.array_equal(a, b)
    other = {**cfg, "seed": cfg["seed"] + 1}
    c, _ = make_split(model_table, other)
    assert not np.array_equal(a, c)


def test_split_is_at_loan_level(cfg, model_table):
    dev, test = make_split(model_table, cfg)
    ids_dev, ids_test = set(model_table.loan_id.iloc[dev]), set(model_table.loan_id.iloc[test])
    assert model_table.loan_id.is_unique and ids_dev.isdisjoint(ids_test)
