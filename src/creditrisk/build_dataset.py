"""Build the modeling table, the LGD table, and the data funnel.

Usage (from repo root):  py src/creditrisk/build_dataset.py
Reads configs/project.yaml + configs/schema_map.yaml, writes:
  data/processed/model_table.parquet    one row per eligible loan: features, target, EAD, meta_* columns
  data/processed/liquidations.parquet   credit-event terminations with realized loss (for LGD)
  data/processed/feature_spec.json      numeric / categorical feature lists
  reports/data_funnel.csv               loan counts at each step
Columns starting with meta_ are for reporting only and must never be used as features.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/project.yaml")
ap.add_argument("--schema", default="configs/schema_map.yaml")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
schema = yaml.safe_load(open(args.schema))
omap = {c["pos"]: c["name"] for c in schema["origination"]}
pmap = {c["pos"]: c["name"] for c in schema["performance"]}

vy = cfg["vintage_year"]
raw = Path(cfg["paths"]["raw"])
orig_path, perf_path = raw / f"sample_orig_{vy}.txt", raw / f"sample_perf_{vy}.txt"
AGE = cfg["prediction_rule"]["age_months"]
W = cfg["outcome_window_months"]
de = cfg["default_event"]
CREDIT, EXCL, CENS = de["credit_zero_balance_codes"], de["excluded_termination_codes"], de["censored_codes"]

NUM_ORIG = ["credit_score", "ltv", "cltv", "dti", "orig_upb", "orig_rate", "mi_pct",
            "num_units", "orig_term", "num_borrowers"]
CAT_ORIG = ["first_time_homebuyer", "occupancy", "channel", "loan_purpose", "property_type",
            "state", "prepay_penalty_flag", "super_conforming", "harp_flag"]
NUM_PRED = ["upb_ratio", "delinq_at_pred", "max_delinq_prior12", "ever_90_before"]
CAT_PRED = ["mod_at_pred"]

funnel = []


def step(label, n):
    funnel.append({"step": label, "loans": int(n)})
    print(f"{label:<58}{int(n):>9,}")


def read(path, mapping, names=None):
    pos = sorted(p for p in mapping if names is None or mapping[p] in names)
    df = pd.read_csv(path, sep="|", header=None, dtype=str, usecols=pos,
                     keep_default_na=False, na_values=[""], low_memory=False)
    return df.rename(columns=mapping)


def idx(s):  # YYYYMM -> month index
    v = pd.to_numeric(s)
    return (v // 100) * 12 + (v % 100) - 1


# ---------------- origination ----------------
o = read(orig_path, omap)
step("origination loans in sample file", len(o))
for c in schema["origination"]:  # documented "Not Available" codes -> NaN
    if c.get("missing"):
        o.loc[o[c["name"]].isin(c["missing"]), c["name"]] = np.nan
for c in NUM_ORIG:
    o[c] = pd.to_numeric(o[c], errors="coerce")
mx = cfg.get("max_first_payment")
if mx:
    o = o[pd.to_numeric(o.first_payment_date) <= mx].copy()
step(f"after dropping first payment later than {mx}", len(o))
o["fp_idx"] = idx(o.first_payment_date)

# ---------------- performance ----------------
need = ["loan_id", "period", "current_upb", "delinq_status", "zero_bal_code", "mod_flag", "actual_loss",
        "zero_bal_removal_upb", "disaster_delinq_flag", "borrower_assist_plan"]
p = read(perf_path, pmap, need)
p = p[p.loan_id.isin(set(o.loan_id))].copy()
p["pidx"] = idx(p.period)
st = p.delinq_status
p["delq_num"] = pd.to_numeric(st.where(st.str.fullmatch(r"\d{2}", na=False)), errors="coerce")
p["sev90"] = (p.delq_num >= de["delinq_status_min"]) | (st == de["reo_status"])
p["credit_zb"] = p.zero_bal_code.isin(CREDIT)
p["event"] = p.sev90 | p.credit_zb
p["status_ok"] = st.isin(["00", "01", "02"])
p["upb"] = pd.to_numeric(p.current_upb, errors="coerce")

# ---------------- prediction snapshot and eligibility ----------------
pred = o[["loan_id", "fp_idx"]].assign(pred=lambda d: d.fp_idx + AGE - 1)[["loan_id", "pred"]]
snap = pred.merge(p[["loan_id", "pidx", "status_ok", "zero_bal_code", "upb", "delq_num", "mod_flag"]],
                  left_on=["loan_id", "pred"], right_on=["loan_id", "pidx"], how="inner")
step(f"with a performance row at age {AGE}", len(snap))
elig = snap[snap.status_ok & snap.zero_bal_code.isna() & (snap.upb > 0)].copy()
step("eligible: status < 90 days, active, balance > 0", len(elig))

# ---------------- target (events strictly AFTER the prediction month) ----------------
ev = p.loc[p.event, ["loan_id", "pidx", "sev90", "disaster_delinq_flag", "borrower_assist_plan"]]
ev = ev.rename(columns={"pidx": "ev_idx"})
e = elig[["loan_id", "pred"]].merge(ev, on="loan_id", how="left")
e["in_win"] = (e.ev_idx > e.pred) & (e.ev_idx <= e.pred + W)
e["win_dis"] = e.in_win & e.disaster_delinq_flag.eq("Y")
e["win_assist"] = e.in_win & e.borrower_assist_plan.notna()
e["before90"] = (e.ev_idx <= e.pred) & e.sev90.fillna(False).astype(bool)
g = e.groupby("loan_id").agg(target=("in_win", "any"), meta_disaster_event=("win_dis", "any"),
                             meta_assist_event=("win_assist", "any"), ever_90_before=("before90", "any"))

term = (p.loc[p.zero_bal_code.notna(), ["loan_id", "pidx", "zero_bal_code"]]
        .sort_values("pidx").drop_duplicates("loan_id", keep="last")
        .rename(columns={"pidx": "term_idx", "zero_bal_code": "term_code"}).set_index("loan_id"))
d = elig.set_index("loan_id")[["pred", "upb", "delq_num", "mod_flag"]].join(g).join(term)
in_tw = (d.term_idx > d.pred) & (d.term_idx <= d.pred + W)
excluded = in_tw & d.term_code.isin(EXCL) & ~d.target
step("excluded: ended in window as code 16 / 96", excluded.sum())
d = d[~excluded].copy()
prepaid = (in_tw & d.term_code.isin(CENS) & ~d.target).reindex(d.index).fillna(False)
step("  (kept) paid off inside window, counted as non-default", prepaid.sum())
step("MODELING POPULATION", len(d))
step("  defaults in window (target = 1)", d.target.sum())

# ---------------- features known at or before the prediction month ----------------
d = d.join(o.set_index("loan_id")[NUM_ORIG + CAT_ORIG])
d["upb_ratio"] = d.upb / d.orig_upb
d["delinq_at_pred"] = d.delq_num
d["mod_at_pred"] = np.where(d.mod_flag.notna(), "Y", "N")
d["ever_90_before"] = d.ever_90_before.astype(int)
hist = p[["loan_id", "pidx", "delq_num"]].merge(pred, on="loan_id")
hist = hist[(hist.pidx <= hist.pred) & (hist.pidx > hist.pred - 12)]  # prediction month and 11 before
d["max_delinq_prior12"] = hist.groupby("loan_id").delq_num.max().reindex(d.index)

d["ead"] = d.upb  # current actual UPB at the prediction month (USD)
d["pred_period"] = (d.pred // 12).astype(str) + (d.pred % 12 + 1).astype(str).str.zfill(2)
d["target"] = d.target.astype(int)
d["meta_disaster_event"] = d.meta_disaster_event.astype(int)
d["meta_assist_event"] = d.meta_assist_event.astype(int)
model = d.reset_index()[["loan_id", "pred_period", "target", "ead"] + NUM_ORIG + CAT_ORIG + NUM_PRED + CAT_PRED
                        + ["meta_disaster_event", "meta_assist_event"]]

# ---------------- leakage guard ----------------
outcome_names = {c["name"] for c in schema["origination"] + schema["performance"] if c["known_at"] == "outcome"}
features = NUM_ORIG + CAT_ORIG + NUM_PRED + CAT_PRED
assert not (set(features) & outcome_names), "outcome field used as feature"
assert not any(f.startswith("meta_") or f in ("target", "ead", "loan_id") for f in features)

# ---------------- LGD table (whole sample, any date) ----------------
z = (p[p.credit_zb].sort_values("pidx").drop_duplicates("loan_id", keep="last")
     [["loan_id", "zero_bal_code", "pidx", "actual_loss", "zero_bal_removal_upb"]].copy())
z["actual_loss"] = pd.to_numeric(z.actual_loss, errors="coerce")
z["zero_bal_removal_upb"] = pd.to_numeric(z.zero_bal_removal_upb, errors="coerce")
z["lgd_raw"] = z.actual_loss / z.zero_bal_removal_upb
z = z.merge(o[["loan_id", "ltv", "orig_upb", "state", "loan_purpose", "property_type"]], on="loan_id")
z["in_model_population"] = z.loan_id.isin(set(model.loan_id))

# ---------------- save ----------------
out = Path(cfg["paths"]["processed"])
out.mkdir(parents=True, exist_ok=True)
Path("reports").mkdir(exist_ok=True)
model.to_parquet(out / "model_table.parquet", index=False)
z.to_parquet(out / "liquidations.parquet", index=False)
json.dump({"numeric": NUM_ORIG + NUM_PRED, "categorical": CAT_ORIG + CAT_PRED},
          open(out / "feature_spec.json", "w"), indent=2)
pd.DataFrame(funnel).to_csv("reports/data_funnel.csv", index=False)

# ---------------- summary ----------------
print(f"\nDefault rate: {model.target.mean():.3%}  ({model.target.sum()} of {len(model):,})")
print(f"Prediction months: {model.pred_period.min()} to {model.pred_period.max()}")
print(f"Window defaults flagged disaster-related: {int(model[model.target == 1].meta_disaster_event.sum())}; "
      f"in a borrower assistance plan: {int(model[model.target == 1].meta_assist_event.sum())}")
print("\nMissing share of features (top 8):")
print(model[features].isna().mean().sort_values(ascending=False).head(8).round(3).to_string())
print(f"\nCredit-event terminations: {len(z)}; with non-null actual loss: {int(z.actual_loss.notna().sum())}")
print("by zero balance code:", z.zero_bal_code.value_counts().to_dict())
print("lgd_raw quantiles (loss / balance at removal):")
print(z.lgd_raw.describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_string())
print(f"lgd_raw < 0: {(z.lgd_raw < 0).sum()}   > 1: {(z.lgd_raw > 1).sum()}")
print("\nWrote model_table.parquet, liquidations.parquet, feature_spec.json, reports/data_funnel.csv")
