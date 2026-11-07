import copy
from pathlib import Path

import pytest
from synth import make_loans, write_raw

from creditrisk.build_dataset import build
from creditrisk.config import load_config, load_schema

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg():
    """Project config with the table-size floors relaxed: the synthetic fixtures hold a few dozen loans."""
    c = load_config(ROOT / "configs" / "project.yaml")
    c["validation"]["tables"]["origination"]["min_rows"] = 1
    c["validation"]["tables"]["model_table"].update(min_rows=1, min_defaults=0, default_rate=[0.0, 1.0])
    return c


@pytest.fixture(scope="session")
def schema():
    return load_schema(ROOT / "configs" / "schema_map.yaml")


@pytest.fixture
def cfg_copy(cfg):
    return copy.deepcopy(cfg)


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory, cfg, schema):
    loans, orig, perf = make_loans()
    d = tmp_path_factory.mktemp("raw")
    o_path, p_path = write_raw(d, schema, orig, perf)
    model, liq, funnel = build(cfg, schema, o_path, p_path)
    return {"loans": loans, "orig": orig, "perf": perf, "dir": d, "paths": (o_path, p_path),
            "model": model, "liq": liq, "funnel": funnel}


@pytest.fixture(scope="session")
def model_table(cfg):
    from synth import make_model_table

    return make_model_table(cfg)


@pytest.fixture(scope="session")
def fitted_scorecard(cfg, model_table):
    from sklearn.model_selection import StratifiedKFold

    from creditrisk.config import feature_lists
    from creditrisk.train import fit_scorecard

    small = copy.deepcopy(cfg)
    small["scorecard"]["c_grid"] = [0.1, 1.0]
    spec = feature_lists(small, "full")
    y = model_table.target.to_numpy()
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    search = fit_scorecard(model_table, y, spec, small, cv)
    return search.best_estimator_, spec


@pytest.fixture(scope="session")
def serving_table(cfg):
    from synth import make_valid_table

    return make_valid_table(cfg, n=3000, seed=5)


@pytest.fixture(scope="session")
def model_dirs(cfg, serving_table, tmp_path_factory):
    """Exported model folders (scorecard and lgbm, full feature set) trained on a small synthetic table."""
    from creditrisk.config import feature_lists
    from creditrisk.export import write_model_dir
    from creditrisk.features import category_levels
    from creditrisk.metrics import fit_platt
    from creditrisk.model import CalibratedScorer, build_lgbm_pipeline, build_scorecard_pipeline

    dev = serving_table.iloc[:2000].reset_index(drop=True)
    y = dev.target.to_numpy()
    spec = feature_lists(cfg, "full")
    small = copy.deepcopy(cfg)
    small["lgbm"]["fixed"]["n_estimators"] = 30
    params = {k: v[0] for k, v in small["lgbm"]["grid"].items()}
    pipes = {"scorecard": build_scorecard_pipeline(spec, small),
             "lgbm": build_lgbm_pipeline(spec, small, category_levels(dev, spec), params=params)}
    root = tmp_path_factory.mktemp("models")
    out = {}
    for kind, pipe in pipes.items():
        pipe.fit(dev, y)
        eps = cfg["evaluation"]["prob_clip"]
        platt = fit_platt(pipe.predict_proba(dev)[:, 1], y, eps, cfg["evaluation"]["platt_c"])
        scorer = CalibratedScorer(pipe, platt, eps)
        meta = {"source": "test", "registry_name": f"creditrisk-full-{kind}", "registry_version": 1, "run_id": "r1"}
        write_model_dir(scorer, cfg, ROOT / "configs" / "project.yaml", root / kind, meta)
        out[kind] = root / kind
    return out
