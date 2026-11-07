"""Export one model for serving: a self-contained folder the API and batch scorer load (and Docker bakes in).

Usage (from repo root):
  py -m creditrisk.export --name creditrisk-full-scorecard --alias production --out deploy/model
  py -m creditrisk.export --local artifacts/full/scorecard.joblib --out deploy/model
Writes scorer.joblib, project.yaml (the configuration the model was exported with, including decision cutoffs and
reason codes) and metadata.json (registry name and version, run, git commit, data fingerprint, SHA-256 of the scorer).
The server refuses to start if scorer.joblib no longer matches its recorded checksum.
"""
import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import joblib

from creditrisk.config import common_args, load_config
from creditrisk.lineage import sha256_file
from creditrisk.serving import CONFIG, METADATA, SCORER, model_kind


def feature_set_of(cfg, scorer):
    behaviour = set(cfg["features"]["behaviour"]["numeric"]) | set(cfg["features"]["behaviour"]["categorical"])
    names = set(scorer.spec["numeric"]) | set(scorer.spec["categorical"])
    return "full" if names & behaviour else "origination"


def write_model_dir(scorer, cfg, config_path, out, extra=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(scorer, out / SCORER)
    shutil.copyfile(config_path, out / CONFIG)
    meta = {"feature_set": feature_set_of(cfg, scorer), "model_kind": model_kind(scorer),
            "scorer_sha256": sha256_file(out / SCORER), "config_sha256": sha256_file(out / CONFIG),
            "exported_at_utc": datetime.now(UTC).isoformat(timespec="seconds"), **(extra or {})}
    (out / METADATA).write_text(json.dumps(meta, indent=2))
    return meta


def from_registry(cfg, name, version, alias):
    import mlflow

    from creditrisk.registry import get_version, record
    from creditrisk.tracking import setup

    client = setup(cfg)
    mv = get_version(client, name, version, alias)
    rec = record(client, mv)
    scorer = mlflow.sklearn.load_model(f"models:/{name}/{mv.version}")
    t = rec["tags"]
    extra = {"source": "registry", "registry_name": name, "registry_version": int(mv.version),
             "registry_aliases": rec["aliases"], "run_id": rec["run_id"], "git_commit": t.get("git_commit"),
             "code_dirty": t.get("code_dirty"), "data_fingerprint": t.get("data_fingerprint"),
             "registry_config_sha256": t.get("config_sha256"),
             "test_roc_auc": rec["metrics"].get(f"test.{rec['model_kind']}_platt.roc_auc")}
    return scorer, extra


def main(argv=None):
    ap = common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--out", default="deploy/model")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--name", help="registered model name")
    src.add_argument("--local", help="path to a scorer saved by creditrisk.train")
    ap.add_argument("--version", type=int)
    ap.add_argument("--alias", default=None, help="registry alias, e.g. production or candidate")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.local:
        scorer, extra = joblib.load(args.local), {"source": "local", "local_path": str(args.local)}
    else:
        if (args.version is None) == (args.alias is None):
            raise SystemExit("give exactly one of --version or --alias with --name")
        scorer, extra = from_registry(cfg, args.name, args.version, args.alias)
    meta = write_model_dir(scorer, cfg, args.config, args.out, extra)
    print(f"Exported {meta.get('registry_name') or args.local} ({meta['model_kind']}, {meta['feature_set']} set, "
          f"registry version {meta.get('registry_version', 'n/a')}) to {args.out}")


if __name__ == "__main__":
    main()
