"""Model registry commands: list versions, answer "what produced this model?", promote, roll back.

Usage (from repo root):
  py -m creditrisk.registry list
  py -m creditrisk.registry lineage --name creditrisk-full-lgbm --alias candidate [--check] [--json]
  py -m creditrisk.registry promote --name creditrisk-full-lgbm --version 3 [--force --reason "why"]
  py -m creditrisk.registry rollback --name creditrisk-full-lgbm

Labels are registry aliases: 'candidate' is set automatically on every newly trained version, 'production' only by
`promote`, after the gates in configs/project.yaml (mlflow.promotion) pass. The previously promoted version keeps the
'previous_production' alias so `rollback` can restore it. Every promotion is tagged with who, when and any override.
"""
import argparse
import getpass
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from creditrisk.config import common_args, load_config
from creditrisk.lineage import git_info, sha256_file, table_fingerprint
from creditrisk.tracking import TAG, read_text_artifact, setup


class PromotionError(RuntimeError):
    pass


def get_version(client, name, version=None, alias=None):
    if alias:
        return client.get_model_version_by_alias(name, alias)
    if version is None:
        raise ValueError("give --version or --alias")
    return client.get_model_version(name, str(version))


def record(client, mv):
    """Everything known about one registered version: its run's tags, parameters and metrics plus the lineage file."""
    run = client.get_run(mv.run_id)
    tags = {k.removeprefix(TAG): v for k, v in run.data.tags.items() if k.startswith(TAG)}
    try:
        lineage = json.loads(read_text_artifact(mv.run_id, "lineage/lineage.json"))
    except Exception:
        lineage = None
    return {"name": mv.name, "version": int(mv.version), "aliases": sorted(mv.aliases), "run_id": mv.run_id,
            "started_utc": datetime.fromtimestamp(run.info.start_time / 1000, UTC).isoformat(timespec="seconds"),
            "model_kind": mv.tags.get(f"{TAG}model_kind"), "tags": tags, "params": dict(run.data.params),
            "metrics": dict(run.data.metrics), "lineage": lineage, "version_tags": dict(mv.tags)}


def list_versions(client):
    rows = []
    for rm in sorted(client.search_registered_models(), key=lambda m: m.name):
        for found in sorted(client.search_model_versions(f"name='{rm.name}'"), key=lambda v: -int(v.version)):
            mv = client.get_model_version(rm.name, found.version)  # a search result does not carry the aliases
            t = mv.tags
            rows.append({"model": mv.name, "version": int(mv.version), "aliases": ",".join(sorted(mv.aliases)),
                         "test_roc_auc": t.get(f"{TAG}test_roc_auc"), "code_dirty": t.get(f"{TAG}code_dirty"),
                         "commit": (t.get(f"{TAG}git_commit") or "")[:8],
                         "data": (t.get(f"{TAG}data_fingerprint") or "")[:8],
                         "run": mv.run_id[:8]})
    return pd.DataFrame(rows)


def describe(rec):
    t, p, m = rec["tags"], rec["params"], rec["metrics"]
    lin = rec["lineage"] or {}
    manifest = (lin.get("data") or {}).get("manifest") or {}
    raw = manifest.get("raw", {})
    table = manifest.get("model_table", {})
    kind = rec["model_kind"]
    chosen = {k.split("chosen.")[1]: v for k, v in p.items() if k.startswith(f"{kind}.chosen.")}
    lines = [f"MODEL {rec['name']} version {rec['version']}   aliases: {', '.join(rec['aliases']) or '-'}",
             f"trained in run {rec['run_id']} at {rec['started_utc']} UTC", "", "DATA"]
    for label, key in (("raw origination", "origination"), ("raw performance", "performance")):
        e = raw.get(key, {})
        lines.append(f"  {label:<18}{e.get('file', '?')}  sha256 {e.get('sha256', t.get('raw_orig_sha256', '?'))}")
    lines += [f"  modeling table    {table.get('rows', '?')} loans, {table.get('defaults', '?')} defaults, "
              f"fingerprint {t.get('data_fingerprint')}",
              f"  built from the recorded raw files: {t.get('data_matches_manifest')}", "", "CODE",
              f"  git commit {t.get('git_commit')} on {t.get('git_branch')}",
              f"  uncommitted changes in code paths when trained: {t.get('code_dirty')}"]
    dirty = (lin.get("code") or {}).get("dirty_files")
    if dirty:
        lines.append(f"  changed files: {', '.join(dirty)} (diff stored with the run: lineage/code_uncommitted.diff)")
    lines += ["", "PARAMETERS", f"  configuration sha256 {t.get('config_sha256')}  (full file stored with the run)",
              f"  seed {p.get('seed')}, test share {p.get('split.test_size')}, cv folds {p.get('split.cv_folds')}",
              f"  chosen by cross-validation: {chosen}",
              f"  features: {p.get('n_features')} ({p.get('feature_set')} set)"]
    env = lin.get("environment") or {}
    packages = ", ".join(f"{k} {v}" for k, v in (env.get("packages") or {}).items())
    lines += ["", "ENVIRONMENT",
              f"  python {env.get('python', '?')}; requirements.txt sha256 {t.get('requirements_sha256')}",
              f"  {packages}", "", "TEST RESULTS (Platt-calibrated)"]
    for metric in ("roc_auc", "pr_auc", "brier", "recall_top10"):
        k = f"test.{kind}_platt.{metric}"
        if k in m:
            lines.append(f"  {metric:<14}{m[k]:.4f}  (95% CI {m[k + '.ci_low']:.4f} to {m[k + '.ci_high']:.4f})")
    return "\n".join(lines)


def check_current(rec, cfg):
    """Compare a version's recorded lineage with this checkout: [(item, status, detail)]."""
    t = rec["tags"]
    ml = cfg["mlflow"]
    out = []

    def add(item, recorded, current):
        status = "n/a" if current is None else "MATCH" if recorded == current else "DIFFERS"
        out.append((item, status, f"recorded {str(recorded)[:12]}, now {str(current)[:12]}"))

    cfg_path = Path("configs/project.yaml")
    add("configuration file", t.get("config_sha256"), sha256_file(cfg_path) if cfg_path.exists() else None)
    table = Path(cfg["paths"]["processed"]) / "model_table.parquet"
    now = table_fingerprint(pd.read_parquet(table)) if table.exists() else None
    add("modeling table", t.get("data_fingerprint"), now)
    vy = cfg["vintage_year"]
    for label, tag, fname in (("raw origination file", "raw_orig_sha256", f"sample_orig_{vy}.txt"),
                              ("raw performance file", "raw_perf_sha256", f"sample_perf_{vy}.txt")):
        f = Path(cfg["paths"]["raw"]) / fname
        add(label, t.get(tag), sha256_file(f) if f.exists() else None)
    g = git_info(ml["code_root"], ml["code_paths"])
    add("git commit", t.get("git_commit"), None if g["commit"] == "unknown" else g["commit"])
    req = Path(ml["code_root"]) / "requirements.txt"
    add("requirements.txt", t.get("requirements_sha256"), sha256_file(req) if req.exists() else None)
    return out


def _auc(client, mv):
    run = client.get_run(mv.run_id)
    return run.data.metrics.get(f"test.{mv.tags.get(f'{TAG}model_kind')}_platt.roc_auc")


def gate_failures(client, cfg, mv):
    """Reasons a version may not be promoted; an empty list means every gate passed."""
    ml = cfg["mlflow"]
    gates = ml["promotion"]
    run = client.get_run(mv.run_id)
    tags = {k.removeprefix(TAG): v for k, v in run.data.tags.items() if k.startswith(TAG)}
    problems = []
    auc = _auc(client, mv)
    if auc is None:
        problems.append("no test ROC-AUC recorded for this version")
    elif auc < gates["min_test_roc_auc"]:
        problems.append(f"test ROC-AUC {auc:.4f} is below the minimum {gates['min_test_roc_auc']}")
    if gates["require_clean_code"] and tags.get("code_dirty") != "false":
        problems.append(f"code state when trained was '{tags.get('code_dirty')}' (commit the code, then retrain)")
    if gates["require_matching_data"] and tags.get("data_matches_manifest") != "true":
        problems.append("the modeling table did not match its data manifest (rerun build_dataset, then retrain)")
    try:
        prod = client.get_model_version_by_alias(mv.name, ml["production_alias"])
    except Exception:
        prod = None
    if prod is not None and prod.version != mv.version and auc is not None:
        prod_auc = _auc(client, prod)
        if prod_auc is not None and auc < prod_auc - gates["max_roc_auc_drop_vs_production"]:
            problems.append(f"test ROC-AUC {auc:.4f} is more than {gates['max_roc_auc_drop_vs_production']} below "
                            f"production version {prod.version} ({prod_auc:.4f})")
    return problems


def promote(client, cfg, name, version, force=False, reason=None):
    ml = cfg["mlflow"]
    mv = get_version(client, name, version)
    problems = gate_failures(client, cfg, mv)
    if problems and not (force and reason):
        hint = "" if force else "\nTo override, pass --force --reason \"...\" (the override is recorded)."
        raise PromotionError("promotion blocked:\n  - " + "\n  - ".join(problems) + hint)
    try:
        current = client.get_model_version_by_alias(name, ml["production_alias"])
    except Exception:
        current = None
    if current is not None and current.version == mv.version:
        return mv, problems
    if current is not None:
        client.set_registered_model_alias(name, ml["previous_alias"], current.version)
    client.set_registered_model_alias(name, ml["production_alias"], mv.version)
    client.set_model_version_tag(name, mv.version, f"{TAG}promoted_by", getpass.getuser())
    client.set_model_version_tag(name, mv.version, f"{TAG}promoted_at_utc",
                                 datetime.now(UTC).isoformat(timespec="seconds"))
    if problems:
        client.set_model_version_tag(name, mv.version, f"{TAG}promotion_override",
                                     f"{reason} | gates failed: {'; '.join(problems)}")
    return mv, problems


def rollback(client, cfg, name):
    ml = cfg["mlflow"]
    try:
        previous = client.get_model_version_by_alias(name, ml["previous_alias"])
    except Exception:
        raise PromotionError(f"{name} has no '{ml['previous_alias']}' version to roll back to") from None
    client.set_registered_model_alias(name, ml["production_alias"], previous.version)
    client.delete_registered_model_alias(name, ml["previous_alias"])
    client.set_model_version_tag(name, previous.version, f"{TAG}rolled_back_at_utc",
                                 datetime.now(UTC).isoformat(timespec="seconds"))
    return previous


def main(argv=None):
    parent = common_args(argparse.ArgumentParser(add_help=False))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list", parents=[parent], help="all registered versions with aliases and headline metrics")
    for cmd in ("lineage", "promote", "rollback"):
        p = sub.add_parser(cmd, parents=[parent])
        p.add_argument("--name", required=True)
        if cmd != "rollback":
            p.add_argument("--version", type=int)
        if cmd == "lineage":
            p.add_argument("--alias")
            p.add_argument("--check", action="store_true", help="compare the recorded lineage with this checkout")
            p.add_argument("--json", action="store_true")
        if cmd == "promote":
            p.add_argument("--force", action="store_true")
            p.add_argument("--reason")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    client = setup(cfg)
    if args.command == "list":
        table = list_versions(client)
        print(table.to_string(index=False) if len(table) else "no registered models yet; run creditrisk.train")
    elif args.command == "lineage":
        rec = record(client, get_version(client, args.name, args.version, args.alias))
        print(json.dumps(rec, indent=2, default=str) if args.json else describe(rec))
        if args.check:
            print("\nCOMPARED WITH THIS CHECKOUT")
            for item, status, detail in check_current(rec, cfg):
                print(f"  {item:<22}{status:<9}{detail}")
    elif args.command == "promote":
        if args.force and not args.reason:
            raise SystemExit("--force needs --reason so the override is explained in the registry")
        try:
            mv, overridden = promote(client, cfg, args.name, args.version, args.force, args.reason)
        except PromotionError as err:
            raise SystemExit(str(err)) from None
        note = f" (OVERRIDE: {len(overridden)} gate(s) failed)" if overridden else ""
        print(f"{args.name} version {mv.version} is now '{cfg['mlflow']['production_alias']}'{note}")
    else:
        try:
            mv = rollback(client, cfg, args.name)
        except PromotionError as err:
            raise SystemExit(str(err)) from None
        print(f"{args.name}: '{cfg['mlflow']['production_alias']}' restored to version {mv.version}")


if __name__ == "__main__":
    main()
