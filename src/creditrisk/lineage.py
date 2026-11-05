"""Provenance: which data, code, configuration and environment produced a model.

Every value here is a hash or a version string that can be recomputed later, so a registered model can be compared
with the current checkout:
  data         SHA-256 of the raw Freddie Mac files and of the cleaned modeling table (data_manifest.json, written by
               build_dataset), plus a content fingerprint of the modeling table as it was when training started
  code         git commit and whether the code paths (config mlflow.code_paths) had uncommitted changes, with the diff
  config       SHA-256 of the whole configs/project.yaml content
  environment  Python, key package versions and the SHA-256 of requirements.txt
"""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import pandas as pd

PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "pandera", "mlflow", "creditrisk")
UNKNOWN = "unknown"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def table_fingerprint(df):
    """Content hash of a table: column names plus a hash of every row, sensitive to values and row order."""
    h = hashlib.sha256()
    h.update(json.dumps(list(df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


def _git(root, *args):
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=60)
    if out.returncode:
        raise RuntimeError(out.stderr.strip())
    return out.stdout


def git_info(root, code_paths):
    """Commit, branch and uncommitted changes under code_paths. Everything is 'unknown' outside a git repository."""
    try:
        commit = _git(root, "rev-parse", "HEAD").strip()
        branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
        status = _git(root, "status", "--porcelain", "--", *code_paths)
        files = [line[3:] for line in status.splitlines()]
        diff = _git(root, "diff", "HEAD", "--", *code_paths) if files else ""
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return {"commit": UNKNOWN, "branch": UNKNOWN, "dirty": UNKNOWN, "dirty_files": [], "diff": ""}
    return {"commit": commit, "branch": branch, "dirty": "true" if files else "false",
            "dirty_files": files[:50], "diff": diff}


def environment_info(root):
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = UNKNOWN
    req = Path(root) / "requirements.txt"
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": versions,
            "requirements_sha256": sha256_file(req) if req.exists() else UNKNOWN}


def _file_entry(path):
    path = Path(path)
    return {"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def data_manifest(cfg, orig_path, perf_path, schema_path, table_path):
    """Record written next to the modeling table by build_dataset: the exact inputs it was built from."""
    table = pd.read_parquet(table_path)
    ml = cfg.get("mlflow", {})
    return {
        "built_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "raw": {"origination": _file_entry(orig_path), "performance": _file_entry(perf_path)},
        "schema_map_sha256": sha256_file(schema_path),
        "rules_sha256": sha256_json({k: cfg[k] for k in ("vintage_year", "prediction_rule", "outcome_window_months",
                                                        "max_first_payment", "default_event", "features")}),
        "model_table": {"rows": len(table), "defaults": int(table.target.sum()),
                        "fingerprint": table_fingerprint(table), "file_sha256": sha256_file(table_path)},
        "code": {k: v for k, v in git_info(ml.get("code_root", "."), ml.get("code_paths", ["src"])).items()
                 if k != "diff"},
    }


def collect(cfg, table, config_path="configs/project.yaml"):
    """Lineage record for a training run that starts from this modeling table (a DataFrame as read from disk)."""
    ml = cfg["mlflow"]
    manifest_path = Path(cfg["paths"]["processed"]) / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    fingerprint = table_fingerprint(table)
    git = git_info(ml["code_root"], ml["code_paths"])
    return {
        "data": {"manifest": manifest, "model_table_fingerprint": fingerprint, "manifest_found": manifest is not None,
                 "matches_manifest": bool(manifest) and manifest["model_table"]["fingerprint"] == fingerprint},
        "code": git,
        "config": {"sha256": sha256_file(config_path) if Path(config_path).exists() else sha256_json(cfg),
                   "path": str(config_path)},
        "environment": environment_info(ml["code_root"]),
    }
