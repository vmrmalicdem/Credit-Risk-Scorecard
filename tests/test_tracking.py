"""End to end: train on a small synthetic table, then answer 'what produced this model?' from the registry."""
import copy
import json
import shutil
import subprocess

import mlflow
import numpy as np
import pandas as pd
import pytest
from synth import make_valid_table

from creditrisk import registry
from creditrisk.lineage import data_manifest
from creditrisk.tracking import TAG, setup
from creditrisk.train import run

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
SCORECARD, LGBM = "creditrisk-full-scorecard", "creditrisk-full-lgbm"


def git(root, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture(scope="module")
def env(tmp_path_factory, cfg):
    root = tmp_path_factory.mktemp("repo")
    (root / "src").mkdir()
    (root / "src" / "a.py").write_text("x = 1\n")
    (root / "configs").mkdir()
    (root / "requirements.txt").write_text("numpy==1.0\n")
    c = copy.deepcopy(cfg)
    c["paths"] = {k: str(root / v) for k, v in c["paths"].items()}
    c["scorecard"]["c_grid"] = [0.1, 1.0]
    c["lgbm"]["fixed"]["n_estimators"] = 20
    c["lgbm"]["grid"] = {"learning_rate": [0.1], "num_leaves": [4], "min_child_samples": [50], "reg_lambda": [1]}
    c["evaluation"]["n_bootstrap"] = 20
    c["mlflow"].update(tracking_uri=f"sqlite:///{(root / 'mlflow.db').as_posix()}", artifact_root=str(root / "runs"),
                       code_root=str(root), code_paths=["src", "configs", "requirements.txt"])
    c["mlflow"]["promotion"].update(min_test_roc_auc=0.5, max_roc_auc_drop_vs_production=0.05)
    cfg_path = root / "configs" / "project.yaml"
    cfg_path.write_text("seed: 42\n")
    proc = root / "data" / "processed"
    proc.mkdir(parents=True)
    table = make_valid_table(c, n=3000)
    table.to_parquet(proc / "model_table.parquet", index=False)
    (root / "data" / "raw").mkdir()
    for name in ("sample_orig_2015.txt", "sample_perf_2015.txt"):
        (root / "data" / "raw" / name).write_text(name)
    (root / "schema.yaml").write_text("origination: []\n")
    manifest = data_manifest(c, root / "data/raw/sample_orig_2015.txt", root / "data/raw/sample_perf_2015.txt",
                             root / "schema.yaml", proc / "model_table.parquet")
    (proc / "data_manifest.json").write_text(json.dumps(manifest))
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "init")
    return {"root": root, "cfg": c, "cfg_path": cfg_path, "table": table}


@pytest.fixture(scope="module")
def first(env):
    result = run(env["cfg"], "full", log=lambda *_: None, config_path=env["cfg_path"])
    return {**env, "result": result, "client": setup(env["cfg"])}


def test_training_registers_both_models_as_candidates(first):
    versions = first["result"]["mlflow"]["versions"]
    assert versions["scorecard"] == {"name": SCORECARD, "version": 1}
    assert versions["lgbm"] == {"name": LGBM, "version": 1}
    for name in (SCORECARD, LGBM):
        mv = first["client"].get_model_version_by_alias(name, "candidate")
        assert int(mv.version) == 1 and "production" not in mv.aliases


def test_run_has_parameters_metrics_tags_and_artifacts(first):
    client = first["client"]
    mv = client.get_model_version_by_alias(LGBM, "candidate")
    run_ = client.get_run(mv.run_id)
    assert run_.data.params["lgbm.chosen.num_leaves"] == "4"
    assert run_.data.params["seed"] == "42" and run_.data.params["scorecard.chosen.C"] in ("0.1", "1.0")
    m = run_.data.metrics
    assert 0.5 < m["test.lgbm_platt.roc_auc"] < 1 and m["test.lgbm_platt.roc_auc.ci_low"] < m["test.lgbm_platt.roc_auc"]
    assert "cv.scorecard.neg_log_loss" in m
    tags = run_.data.tags
    assert tags[f"{TAG}git_commit"] == git(first["root"], "rev-parse", "HEAD")
    assert tags[f"{TAG}code_dirty"] == "false" and tags[f"{TAG}data_matches_manifest"] == "true"
    paths = {a.path for a in client.list_artifacts(mv.run_id)}
    assert {"lineage", "inputs", "reports", "run"} <= paths


def test_registered_model_reproduces_the_saved_test_predictions(first):
    cfg, table = first["cfg"], first["table"]
    split = pd.read_parquet(f"{cfg['paths']['artifacts']}/full/split.parquet").set_index("loan_id").split
    pred = pd.read_parquet(f"{cfg['paths']['artifacts']}/full/predictions.parquet").set_index("loan_id")
    test = table[table.loan_id.map(split) == "test"]
    for kind, name in (("scorecard", SCORECARD), ("lgbm", LGBM)):
        model = mlflow.sklearn.load_model(f"models:/{name}@candidate")
        assert np.allclose(model.predict(test), pred.loc[test.loan_id, f"p_{kind}_platt"].to_numpy(), atol=1e-12)


def test_lineage_answers_data_code_and_parameters(first):
    mv = first["client"].get_model_version_by_alias(LGBM, "candidate")
    rec = registry.record(first["client"], mv)
    text = registry.describe(rec)
    assert git(first["root"], "rev-parse", "HEAD") in text
    raw = rec["lineage"]["data"]["manifest"]["raw"]["origination"]["sha256"]
    assert raw in text and rec["tags"]["data_fingerprint"] in text
    assert "num_leaves" in text and rec["tags"]["config_sha256"] in text and "roc_auc" in text


def test_lineage_check_matches_the_checkout_the_model_was_trained_from(first, monkeypatch):
    monkeypatch.chdir(first["root"])
    cfg = copy.deepcopy(first["cfg"])
    cfg["paths"]["raw"] = str(first["root"] / "data" / "raw")
    rec = registry.record(first["client"], first["client"].get_model_version_by_alias(LGBM, "candidate"))
    status = {item: s for item, s, _ in registry.check_current(rec, cfg)}
    assert status["configuration file"] == status["modeling table"] == status["git commit"] == "MATCH"
    assert status["raw origination file"] == status["requirements.txt"] == "MATCH"
    (first["root"] / "configs" / "project.yaml").write_text("seed: 43\n")
    changed = {item: s for item, s, _ in registry.check_current(rec, cfg)}
    (first["root"] / "configs" / "project.yaml").write_text("seed: 42\n")
    assert changed["configuration file"] == "DIFFERS"


def test_promote_moves_the_production_label_and_supports_rollback(first):
    client, cfg = first["client"], first["cfg"]
    registry.promote(client, cfg, LGBM, 1)
    assert int(client.get_model_version_by_alias(LGBM, "production").version) == 1
    assert client.get_model_version(LGBM, "1").tags[f"{TAG}promoted_by"]
    second = run(cfg, "full", log=lambda *_: None, config_path=first["cfg_path"])["mlflow"]["versions"]["lgbm"]
    assert second["version"] == 2
    assert int(client.get_model_version_by_alias(LGBM, "candidate").version) == 2
    registry.promote(client, cfg, LGBM, 2)
    assert int(client.get_model_version_by_alias(LGBM, "production").version) == 2
    assert int(client.get_model_version_by_alias(LGBM, "previous_production").version) == 1
    registry.rollback(client, cfg, LGBM)
    assert int(client.get_model_version_by_alias(LGBM, "production").version) == 1


def test_rollback_without_a_previous_version_is_an_error(first):
    with pytest.raises(registry.PromotionError):
        registry.rollback(first["client"], first["cfg"], SCORECARD)


def test_gates_block_dirty_code_and_an_override_is_recorded(first):
    client, cfg = first["client"], first["cfg"]
    (first["root"] / "src" / "a.py").write_text("x = 99\n")
    try:
        version = run(cfg, "origination", log=lambda *_: None, config_path=first["cfg_path"])["mlflow"]["versions"]
    finally:
        git(first["root"], "checkout", "--", "src/a.py")
    name = version["scorecard"]["name"]
    run_id = client.get_model_version(name, "1").run_id
    assert "code_uncommitted.diff" in {a.path.split("/")[-1] for a in client.list_artifacts(run_id, "lineage")}
    with pytest.raises(registry.PromotionError, match="code state"):
        registry.promote(client, cfg, name, 1)
    with pytest.raises(registry.PromotionError):
        registry.promote(client, cfg, name, 1, force=True, reason=None)
    registry.promote(client, cfg, name, 1, force=True, reason="demo override")
    tags = client.get_model_version(name, "1").tags
    assert "demo override" in tags[f"{TAG}promotion_override"]
    assert int(client.get_model_version_by_alias(name, "production").version) == 1


def test_gates_block_a_worse_model_and_a_low_score(first):
    client, cfg = first["client"], copy.deepcopy(first["cfg"])
    cfg["mlflow"]["promotion"]["min_test_roc_auc"] = 0.999
    with pytest.raises(registry.PromotionError, match="below the minimum"):
        registry.promote(client, cfg, SCORECARD, 1)
    cfg["mlflow"]["promotion"].update(min_test_roc_auc=0.5, max_roc_auc_drop_vs_production=-1.0)
    registry.promote(client, first["cfg"], SCORECARD, 1)
    run(first["cfg"], "full", log=lambda *_: None, config_path=first["cfg_path"])
    with pytest.raises(registry.PromotionError, match="production version"):
        registry.promote(client, cfg, SCORECARD, 2)


def test_list_shows_every_version_with_its_aliases(first):
    table = registry.list_versions(first["client"])
    assert {SCORECARD, LGBM} <= set(table.model)
    row = table[(table.model == LGBM) & (table.version == 1)].iloc[0]
    assert "production" in row.aliases and row.code_dirty == "false"
