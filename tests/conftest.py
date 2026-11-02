import copy
from pathlib import Path

import pytest
from synth import make_loans, write_raw

from creditrisk.build_dataset import build
from creditrisk.config import load_config, load_schema

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "configs" / "project.yaml")


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
    from creditrisk.features import scorecard_frame
    from creditrisk.train import fit_scorecard

    small = copy.deepcopy(cfg)
    small["scorecard"]["c_grid"] = [0.1, 1.0]
    spec = feature_lists(small, "full")
    y = model_table.target.to_numpy()
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    search = fit_scorecard(scorecard_frame(model_table, spec), y, spec, small, cv)
    return search.best_estimator_, spec
