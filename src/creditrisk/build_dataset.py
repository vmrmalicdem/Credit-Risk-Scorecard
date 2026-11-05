"""Build the modeling table, the LGD table, and the data funnel.

Usage (from repo root):  py -m creditrisk.build_dataset
Reads configs/project.yaml + configs/schema_map.yaml, writes:
  data/processed/model_table.parquet    one row per eligible loan: features, target, EAD, meta_* columns
  data/processed/liquidations.parquet   credit-event terminations with realized loss (for LGD)
  data/processed/feature_spec.json      numeric / categorical feature lists (full set)
  data/processed/data_manifest.json     SHA-256 of the raw files and the modeling table, for model lineage
  reports/data_funnel.csv               loan counts at each step
  reports/validation_{origination,model_table}.csv   observed null rates and ranges next to the configured limits
Columns starting with meta_ are for reporting only and must never be used as features.
"""
import argparse
import json

import numpy as np
import pandas as pd

from creditrisk.config import (
    DEFAULT_SCHEMA,
    common_args,
    feature_lists,
    load_config,
    load_schema,
    project_paths,
)
from creditrisk.leakage import check_feature_config, check_model_features
from creditrisk.lineage import data_manifest
from creditrisk.validation import validate_model_table, validate_origination


def read_table(path, mapping, names=None):
    pos = sorted(p for p in mapping if names is None or mapping[p] in names)
    df = pd.read_csv(path, sep="|", header=None, dtype=str, usecols=pos,
                     keep_default_na=False, na_values=[""], low_memory=False)
    return df.rename(columns=mapping)


def month_index(s):
    """YYYYMM -> consecutive month number."""
    v = pd.to_numeric(s)
    return (v // 100) * 12 + (v % 100) - 1


def build(cfg, schema, orig_path, perf_path, log=None, validation_out=None):
    """Return (model table, liquidations table, funnel DataFrame) from the two raw files.

    The cleaned origination table and the finished modeling table are validated (validation.py). If validation_out
    is a dict it receives the profile tables under "origination" and "model_table".
    """
    omap = {c["pos"]: c["name"] for c in schema["origination"]}
    pmap = {c["pos"]: c["name"] for c in schema["performance"]}
    age, window = cfg["prediction_rule"]["age_months"], cfg["outcome_window_months"]
    de = cfg["default_event"]
    credit, excl, cens = de["credit_zero_balance_codes"], de["excluded_termination_codes"], de["censored_codes"]
    lookback = cfg["features"]["behaviour_lookback_months"]
    orig_feats, pred_feats = cfg["features"]["origination"], cfg["features"]["behaviour"]
    num_orig, cat_orig = orig_feats["numeric"], orig_feats["categorical"]
    num_pred, cat_pred = list(pred_feats["numeric"]), list(pred_feats["categorical"])

    check_feature_config(cfg, schema)
    funnel = []

    def step(label, n):
        funnel.append({"step": label, "loans": int(n)})
        if log:
            log(f"{label:<58}{int(n):>9,}")

    o = read_table(orig_path, omap)
    step("origination loans in sample file", len(o))
    for c in schema["origination"]:
        if c.get("missing"):
            o.loc[o[c["name"]].isin(c["missing"]), c["name"]] = np.nan
    for c in num_orig:
        o[c] = pd.to_numeric(o[c], errors="coerce")
    mx = cfg.get("max_first_payment")
    if mx:
        o = o[pd.to_numeric(o.first_payment_date) <= mx].copy()
    step(f"after dropping first payment later than {mx}", len(o))
    profiles = {"origination": validate_origination(o, cfg)[0]}
    o["fp_idx"] = month_index(o.first_payment_date)

    need = ["loan_id", "period", "current_upb", "delinq_status", "zero_bal_code", "mod_flag", "actual_loss",
            "zero_bal_removal_upb", "disaster_delinq_flag", "borrower_assist_plan"]
    p = read_table(perf_path, pmap, need)
    p = p[p.loan_id.isin(set(o.loan_id))].copy()
    p["pidx"] = month_index(p.period)
    st = p.delinq_status
    p["delq_num"] = pd.to_numeric(st.where(st.str.fullmatch(r"\d{2}", na=False)), errors="coerce")
    p["sev90"] = (p.delq_num >= de["delinq_status_min"]) | (st == de["reo_status"])
    p["credit_zb"] = p.zero_bal_code.isin(credit)
    p["event"] = p.sev90 | p.credit_zb
    p["status_ok"] = st.isin(["00", "01", "02"])
    p["upb"] = pd.to_numeric(p.current_upb, errors="coerce")

    pred = o[["loan_id", "fp_idx"]].assign(pred=lambda d: d.fp_idx + age - 1)[["loan_id", "pred"]]
    snap = pred.merge(p[["loan_id", "pidx", "status_ok", "zero_bal_code", "upb", "delq_num", "mod_flag"]],
                      left_on=["loan_id", "pred"], right_on=["loan_id", "pidx"], how="inner")
    step(f"with a performance row at age {age}", len(snap))
    elig = snap[snap.status_ok & snap.zero_bal_code.isna() & (snap.upb > 0)].copy()
    step("eligible: status < 90 days, active, balance > 0", len(elig))

    # the target uses only events strictly after the prediction month
    ev = p.loc[p.event, ["loan_id", "pidx", "sev90", "disaster_delinq_flag", "borrower_assist_plan"]]
    ev = ev.rename(columns={"pidx": "ev_idx"})
    e = elig[["loan_id", "pred"]].merge(ev, on="loan_id", how="left")
    e["in_win"] = (e.ev_idx > e.pred) & (e.ev_idx <= e.pred + window)
    e["win_dis"] = e.in_win & e.disaster_delinq_flag.eq("Y")
    e["win_assist"] = e.in_win & e.borrower_assist_plan.notna()
    e["before90"] = (e.ev_idx <= e.pred) & e.sev90.fillna(False).astype(bool)
    g = e.groupby("loan_id").agg(target=("in_win", "any"), meta_disaster_event=("win_dis", "any"),
                                 meta_assist_event=("win_assist", "any"), ever_90_before=("before90", "any"))

    term = (p.loc[p.zero_bal_code.notna(), ["loan_id", "pidx", "zero_bal_code"]]
            .sort_values("pidx").drop_duplicates("loan_id", keep="last")
            .rename(columns={"pidx": "term_idx", "zero_bal_code": "term_code"}).set_index("loan_id"))
    d = elig.set_index("loan_id")[["pred", "upb", "delq_num", "mod_flag"]].join(g).join(term)
    in_tw = (d.term_idx > d.pred) & (d.term_idx <= d.pred + window)
    excluded = in_tw & d.term_code.isin(excl) & ~d.target
    step("excluded: ended in window as code 16 / 96", excluded.sum())
    d = d[~excluded].copy()
    prepaid = (in_tw & d.term_code.isin(cens) & ~d.target).reindex(d.index).fillna(False)
    step("  (kept) paid off inside window, counted as non-default", prepaid.sum())
    step("MODELING POPULATION", len(d))
    step("  defaults in window (target = 1)", d.target.sum())

    # features known at or before the prediction month
    d = d.join(o.set_index("loan_id")[num_orig + cat_orig])
    d["upb_ratio"] = d.upb / d.orig_upb
    d["delinq_at_pred"] = d.delq_num
    d["mod_at_pred"] = np.where(d.mod_flag.notna(), "Y", "N")
    d["ever_90_before"] = d.ever_90_before.astype(int)
    hist = p[["loan_id", "pidx", "delq_num"]].merge(pred, on="loan_id")
    hist = hist[(hist.pidx <= hist.pred) & (hist.pidx > hist.pred - lookback)]
    d["max_delinq_prior12"] = hist.groupby("loan_id").delq_num.max().reindex(d.index)

    d["ead"] = d.upb
    d["pred_period"] = (d.pred // 12).astype(str) + (d.pred % 12 + 1).astype(str).str.zfill(2)
    d["target"] = d.target.astype(int)
    d["meta_disaster_event"] = d.meta_disaster_event.astype(int)
    d["meta_assist_event"] = d.meta_assist_event.astype(int)
    features = num_orig + cat_orig + num_pred + cat_pred
    model = d.reset_index()[["loan_id", "pred_period", "target", "ead"] + features
                            + ["meta_disaster_event", "meta_assist_event"]]
    check_model_features(features, schema)
    profiles["model_table"] = validate_model_table(model, cfg)[0]
    if validation_out is not None:
        validation_out.update(profiles)

    z = (p[p.credit_zb].sort_values("pidx").drop_duplicates("loan_id", keep="last")
         [["loan_id", "zero_bal_code", "pidx", "actual_loss", "zero_bal_removal_upb"]].copy())
    z["actual_loss"] = pd.to_numeric(z.actual_loss, errors="coerce")
    z["zero_bal_removal_upb"] = pd.to_numeric(z.zero_bal_removal_upb, errors="coerce")
    z["lgd_raw"] = z.actual_loss / z.zero_bal_removal_upb
    z = z.merge(o[["loan_id", "ltv", "orig_upb", "state", "loan_purpose", "property_type"]], on="loan_id")
    z["in_model_population"] = z.loan_id.isin(set(model.loan_id))
    return model, z, pd.DataFrame(funnel)


def main(argv=None):
    ap = common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--schema", default=DEFAULT_SCHEMA)
    args = ap.parse_args(argv)
    cfg, schema = load_config(args.config), load_schema(args.schema)
    paths = project_paths(cfg)
    vy = cfg["vintage_year"]
    profiles = {}
    model, z, funnel = build(cfg, schema, paths["raw"] / f"sample_orig_{vy}.txt",
                             paths["raw"] / f"sample_perf_{vy}.txt", log=print, validation_out=profiles)

    out = paths["processed"]
    out.mkdir(parents=True, exist_ok=True)
    paths["reports"].mkdir(parents=True, exist_ok=True)
    model.to_parquet(out / "model_table.parquet", index=False)
    z.to_parquet(out / "liquidations.parquet", index=False)
    full = feature_lists(cfg, "full")
    with open(out / "feature_spec.json", "w") as fh:
        json.dump(full, fh, indent=2)
    funnel.to_csv(paths["reports"] / "data_funnel.csv", index=False)
    manifest = data_manifest(cfg, paths["raw"] / f"sample_orig_{vy}.txt", paths["raw"] / f"sample_perf_{vy}.txt",
                             args.schema, out / "model_table.parquet")
    (out / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    for name, table in profiles.items():
        table.to_csv(paths["reports"] / f"validation_{name}.csv", index=False)

    features = full["numeric"] + full["categorical"]
    defaults = model[model.target == 1]
    print(f"\nDefault rate: {model.target.mean():.3%}  ({model.target.sum()} of {len(model):,})")
    print(f"Prediction months: {model.pred_period.min()} to {model.pred_period.max()}")
    print(f"Window defaults flagged disaster-related: {int(defaults.meta_disaster_event.sum())}; "
          f"in a borrower assistance plan: {int(defaults.meta_assist_event.sum())}")
    print("\nMissing share of features (top 8):")
    print(model[features].isna().mean().sort_values(ascending=False).head(8).round(3).to_string())
    print(f"\nCredit-event terminations: {len(z)}; with non-null actual loss: {int(z.actual_loss.notna().sum())}")
    print("by zero balance code:", z.zero_bal_code.value_counts().to_dict())
    print("lgd_raw quantiles (loss / balance at removal):")
    print(z.lgd_raw.describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_string())
    print(f"lgd_raw < 0: {(z.lgd_raw < 0).sum()}   > 1: {(z.lgd_raw > 1).sum()}")
    print("\nWrote model_table.parquet, liquidations.parquet, feature_spec.json, data_manifest.json,"
          " data_funnel.csv, validation_*.csv")


if __name__ == "__main__":
    main()
