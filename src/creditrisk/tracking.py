"""MLflow tracking: one run per training call, with its models registered as new 'candidate' versions.

A run records the parameters (configuration and the values chosen by cross-validation), test metrics with their
bootstrap intervals, the reports, the configuration file, the data manifest and the lineage record (lineage.py).
Each fitted CalibratedScorer is logged as an MLflow model with a signature and registered under
creditrisk-<feature set>-<model>; the new version gets the candidate alias and tags that point back to the data,
code and configuration. The store is local: a SQLite file and an artifact folder under mlruns/ (not committed).
"""
import json
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

from creditrisk.lineage import sha256_json

TAG = "creditrisk."


def registered_name(cfg, feature_set, model):
    return cfg["mlflow"]["model_name"].format(feature_set=feature_set, model=model)


def setup(cfg):
    """Point MLflow at the configured local store and return a client; creates the store and experiment if needed."""
    ml = cfg["mlflow"]
    uri = ml["tracking_uri"]
    if uri.startswith("sqlite:///"):
        Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    root = Path(ml["artifact_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    if client.get_experiment_by_name(ml["experiment"]) is None:
        client.create_experiment(ml["experiment"], artifact_location=root.as_uri())
    mlflow.set_experiment(ml["experiment"])
    return client


def config_params(cfg, feature_set, spec, summary):
    """Flat parameters worth searching and comparing runs by; the full configuration is stored as an artifact."""
    p = {"feature_set": feature_set, "seed": cfg["seed"], "vintage_year": cfg["vintage_year"],
         "prediction_age_months": cfg["prediction_rule"]["age_months"],
         "outcome_window_months": cfg["outcome_window_months"],
         "n_features": len(spec["numeric"]) + len(spec["categorical"]),
         "features_numeric": json.dumps(spec["numeric"]), "features_categorical": json.dumps(spec["categorical"])}
    for k, v in cfg["split"].items():
        p[f"split.{k}"] = v
    for k, v in cfg["scorecard"].items():
        p[f"scorecard.{k}"] = json.dumps(v) if isinstance(v, list) else v
    for k, v in cfg["lgbm"]["fixed"].items():
        p[f"lgbm.fixed.{k}"] = v
    p["lgbm.grid"] = json.dumps(cfg["lgbm"]["grid"])
    for k in ("n_bootstrap", "top_share", "calibration_bins", "prob_clip", "platt_c"):
        p[f"evaluation.{k}"] = cfg["evaluation"][k]
    p.update({f"scorecard.chosen.{k.removeprefix('lr__')}": v for k, v in summary["scorecard_params"].items()})
    p.update({f"lgbm.chosen.{k}": v for k, v in summary["lgbm_params"].items()})
    for k in ("n_dev", "n_test", "events_dev", "events_test"):
        p[k] = summary[k]
    return p


def metric_values(rows, cv_scores):
    """Test metrics as test.<variant>.<metric> with .ci_low / .ci_high, plus the cross-validated tuning score."""
    out = {}
    for r in pd.DataFrame(rows).itertuples():
        base = f"test.{r.model}.{r.metric}"
        out[base], out[f"{base}.ci_low"], out[f"{base}.ci_high"] = r.estimate, r.ci_low, r.ci_high
    out.update({f"cv.{k}.neg_log_loss": v for k, v in cv_scores.items()})
    return out


def lineage_tags(lineage):
    data, code, env = lineage["data"], lineage["code"], lineage["environment"]
    raw = (data["manifest"] or {}).get("raw", {})
    return {
        f"{TAG}git_commit": code["commit"], f"{TAG}git_branch": code["branch"], f"{TAG}code_dirty": code["dirty"],
        f"{TAG}config_sha256": lineage["config"]["sha256"],
        f"{TAG}data_fingerprint": data["model_table_fingerprint"],
        f"{TAG}data_matches_manifest": str(data["matches_manifest"]).lower(),
        f"{TAG}raw_orig_sha256": raw.get("origination", {}).get("sha256", "unknown"),
        f"{TAG}raw_perf_sha256": raw.get("performance", {}).get("sha256", "unknown"),
        f"{TAG}requirements_sha256": env["requirements_sha256"],
    }


def track_training(cfg, feature_set, ctx, log=print):
    """Log one training run and register its models. ctx holds everything train.run produced (see train.py)."""
    ml = cfg["mlflow"]
    client = setup(cfg)
    lineage, summary, art, rep = ctx["lineage"], ctx["summary"], ctx["art"], ctx["rep"]
    versions = {}
    with mlflow.start_run(run_name=f"{feature_set}-train") as run:
        mlflow.log_params(config_params(cfg, feature_set, ctx["spec"], summary))
        mlflow.log_metrics(metric_values(ctx["rows"], ctx["cv_scores"]))
        mlflow.set_tags({**lineage_tags(lineage), f"{TAG}feature_set": feature_set})
        mlflow.log_dict(lineage, "lineage/lineage.json")
        if lineage["code"]["diff"]:
            mlflow.log_text(lineage["code"]["diff"], "lineage/code_uncommitted.diff")
        mlflow.log_artifact(str(ctx["config_path"]), "inputs")
        if lineage["data"]["manifest_found"]:
            mlflow.log_artifact(str(ctx["manifest_path"]), "inputs")
        mlflow.log_artifacts(str(rep), "reports")
        for name in ("run_summary.json", "split.parquet"):
            mlflow.log_artifact(str(art / name), "run")

        cols = ctx["spec"]["numeric"] + ctx["spec"]["categorical"]
        example = ctx["example"][cols]
        example = example.astype(dict.fromkeys(example.select_dtypes("integer").columns, "float64"))
        for kind, scorer in ctx["scorers"].items():
            reg = registered_name(cfg, feature_set, kind)
            signature = infer_signature(example, scorer.predict(example))
            info = mlflow.sklearn.log_model(scorer, name=kind, signature=signature, registered_model_name=reg,
                                           serialization_format="cloudpickle")
            version = str(info.registered_model_version)
            v_tags = {**lineage_tags(lineage), f"{TAG}feature_set": feature_set, f"{TAG}model_kind": kind,
                      f"{TAG}run_id": run.info.run_id,
                      f"{TAG}test_roc_auc": f"{ctx['summary']['test_metrics'][kind + '_platt']['roc_auc']:.4f}",
                      f"{TAG}test_pr_auc": f"{ctx['summary']['test_metrics'][kind + '_platt']['pr_auc']:.4f}",
                      f"{TAG}params_sha256": sha256_json(config_params(cfg, feature_set, ctx["spec"], summary))}
            for k, v in v_tags.items():
                client.set_model_version_tag(reg, version, k, v)
            client.update_model_version(reg, version, f"{feature_set} {kind}; trained by creditrisk.train")
            client.set_registered_model_alias(reg, ml["candidate_alias"], version)
            versions[kind] = {"name": reg, "version": int(version)}
    for v in versions.values():
        log(f"MLflow: registered {v['name']} version {v['version']} as '{ml['candidate_alias']}'"
            f" (run {run.info.run_id[:8]})")
    return {"run_id": run.info.run_id, "versions": versions}


def read_text_artifact(run_id, path):
    with tempfile.TemporaryDirectory() as d:
        return Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path, dst_path=d)).read_text()
