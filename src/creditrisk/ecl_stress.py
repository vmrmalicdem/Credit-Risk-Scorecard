"""LGD segments, simplified 12-month expected loss, and feature-shock stress test.

Usage:  py src/creditrisk/ecl_stress.py
Reads:  data/processed/{model_table,liquidations}.parquet, artifacts/<set>/{scorecard,lgbm}.joblib,
        artifacts/<set>/{split,predictions}.parquet  (both feature sets from train.py must exist)
Writes: reports/lgd_segments.csv, reports/stress_results.csv, artifacts/stress_pd.parquet

LGD: realized loss / balance at removal for credit-event terminations with a reported actual loss, floored at 0
and capped at 1. Default is one pooled average. With --lgd segmented, two segments by origination LTV (above 80
vs 80 or below) are used if each has at least MIN_SEG loans; segment results are written with a _segmented suffix.

ECL (USD, 12 months, simplified, not IFRS 9): per loan PD x LGD x EAD, summed over the portfolio.
  ecl_spec  uses the liquidation-only LGD and treats every default as a liquidation: a conservative upper bound.
  ecl_adj   multiplies by the observed share of in-window defaults that later ended in a liquidation with loss.
  ecl_spec_lgd_reseg  (segmented mode only) re-assigns the LGD segment from the shocked LTV; otherwise LGD is held at baseline.
PD is the Platt-calibrated probability. EAD is current balance at the prediction month.

Scenarios (applied to every loan, separately and combined): LTV and CLTV +10 or +20 percentage points (capped),
original rate +2.0 points, and LTV +20 with rate +2.0. These are sensitivity analyses of fitted models, not
causal estimates or macroeconomic forecasts. Baseline PDs on the test set must reproduce predictions.parquet.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.special import expit, logit

from features import lgbm_frame, scorecard_frame

ap = argparse.ArgumentParser()
ap.add_argument("--lgd", choices=["pooled", "segmented"], default="pooled")
LGD_MODE = ap.parse_args().lgd
cfg = yaml.safe_load(open("configs/project.yaml"))
proc, art, rep = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["artifacts"]), Path("reports")
rep.mkdir(exist_ok=True)
SEED, MIN_SEG, LTV_CAP, LTV_SPLIT = cfg["seed"], 30, 150.0, 80.0
SCEN = {"baseline": (0, 0.0), "ltv_10": (10, 0.0), "ltv_20": (20, 0.0), "rate_200": (0, 2.0), "combined": (20, 2.0)}

df = pd.read_parquet(proc / "model_table.parquet")
liq = pd.read_parquet(proc / "liquidations.parquet")
liq = liq[liq.actual_loss.notna() & (liq.zero_bal_removal_upb > 0)].copy()
liq["lgd"] = liq.lgd_raw.clip(0, 1)
liq["segment"] = np.where(liq.ltv > LTV_SPLIT, "ltv_gt_80", "ltv_le_80")

rng = np.random.default_rng(SEED)


def seg_stats(g, name):
    x = g.lgd.to_numpy()
    boot = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return {"segment": name, "n": len(x), "mean_lgd": x.mean(), "median_lgd": float(np.median(x)),
            "ci_low": np.quantile(boot, 0.025), "ci_high": np.quantile(boot, 0.975),
            "mean_lgd_balance_weighted": np.average(x, weights=g.zero_bal_removal_upb),
            "mean_raw_unclipped": g.lgd_raw.mean(), "n_floored": int((g.lgd_raw < 0).sum()),
            "n_capped": int((g.lgd_raw > 1).sum())}


segs = [seg_stats(g, s) for s, g in liq.groupby("segment")]
use_segments = LGD_MODE == "segmented" and len(segs) == 2 and all(s["n"] >= MIN_SEG for s in segs)
SUF = "_segmented" if use_segments else ""
lgd_tab = pd.DataFrame(segs + [seg_stats(liq, "all")])
lgd_tab.insert(1, "used_in_ecl", [use_segments] * len(segs) + [not use_segments])
lgd_tab.to_csv(rep / "lgd_segments.csv", index=False)
lgd_map = {s["segment"]: s["mean_lgd"] for s in segs}
lgd_all = float(lgd_tab.loc[lgd_tab.segment == "all", "mean_lgd"].iloc[0])


def assign_lgd(ltv):
    if use_segments:
        return np.where(ltv > LTV_SPLIT, lgd_map["ltv_gt_80"], lgd_map["ltv_le_80"])
    return np.full(len(ltv), lgd_all)


defaults = df.loc[df.target == 1, "loan_id"]
conv_n, conv_d = int(defaults.isin(set(liq.loan_id)).sum()), len(defaults)
conv = conv_n / conv_d
z = 1.96
den = 1 + z ** 2 / conv_d
ctr = (conv + z ** 2 / (2 * conv_d)) / den
half = z * np.sqrt(conv * (1 - conv) / conv_d + z ** 2 / (4 * conv_d ** 2)) / den
print(lgd_tab.round(3).to_string(index=False))
print(f"LGD mode: {LGD_MODE}; segments used: {use_segments}")
print(f"in-window defaults that later liquidated with a reported loss: {conv_n}/{conv_d} = {conv:.1%} "
      f"(95% CI {ctr - half:.1%} to {ctr + half:.1%})")


def shock(d, ltv_pp, rate_pts):
    d = d.copy()
    if ltv_pp:
        d["ltv"] = (d.ltv + ltv_pp).clip(upper=LTV_CAP)
        d["cltv"] = (d.cltv + ltv_pp).clip(upper=LTV_CAP)
    if rate_pts:
        d["orig_rate"] = d.orig_rate + rate_pts
    return d


def score(bundle, d, kind):
    spec = bundle["spec"]
    X = scorecard_frame(d, spec) if kind == "scorecard" else lgbm_frame(d, spec, bundle["levels"])
    p = bundle["model"].predict_proba(X)[:, 1]
    a, b = bundle["platt"]
    return expit(a + b * logit(np.clip(p, 1e-6, 1 - 1e-6)))


ltv_ref, rate_ref = float(df.ltv.quantile(0.99)), float(df.orig_rate.quantile(0.99))
lgd_base = assign_lgd(df.ltv.to_numpy())
ead_all = df.ead.to_numpy()
rows, pd_frames = [], []
for fs in ["full", "origination"]:
    split = pd.read_parquet(art / fs / "split.parquet").set_index("loan_id").split
    pred = pd.read_parquet(art / fs / "predictions.parquet").set_index("loan_id")
    is_test = df.loan_id.map(split).eq("test").to_numpy()
    for kind in ["scorecard", "lgbm"]:
        bundle = joblib.load(art / fs / f"{kind}.joblib")
        base = None
        for sname, (lp, rp) in SCEN.items():
            d = shock(df, lp, rp)
            p = score(bundle, d, kind)
            if sname == "baseline":
                assert np.allclose(p[is_test], pred.loc[df.loan_id[is_test], f"p_{kind}_platt"].to_numpy(),
                                   atol=1e-6), f"baseline mismatch for {fs}/{kind}"
                base = p
            lgd_shock = assign_lgd(d.ltv.to_numpy())
            pd_frames.append(pd.DataFrame({"loan_id": df.loan_id, "feature_set": fs, "model": kind,
                                           "scenario": sname, "pd": p}))
            for popn, m in [("test", is_test), ("all", np.ones(len(df), bool))]:
                ead = ead_all[m]
                ecl = float((p[m] * lgd_base[m] * ead).sum())
                rows.append({
                    "feature_set": fs, "model": kind, "population": popn, "scenario": sname,
                    "loans": int(m.sum()), "observed_defaults": int(df.target[m].sum()),
                    "expected_defaults": float(p[m].sum()), "mean_pd": float(p[m].mean()),
                    "mean_delta_pd": float((p[m] - base[m]).mean()), "share_pd_up": float((p[m] > base[m]).mean()),
                    "ead_total": float(ead.sum()), "ecl_spec": ecl, "ecl_spec_rate": ecl / float(ead.sum()),
                    "ecl_adj": ecl * conv,
                    "share_ltv_above_p99": float((d.ltv[m] > ltv_ref).mean()),
                    "share_rate_above_p99": float((d.orig_rate[m] > rate_ref).mean()),
                    **({"ecl_spec_lgd_reseg": float((p[m] * lgd_shock[m] * ead).sum())} if use_segments else {}),
                })

res = pd.DataFrame(rows)
keys = ["feature_set", "model", "population"]
b0 = res[res.scenario == "baseline"].set_index(keys)
res["ecl_spec_pct_vs_base"] = 100 * (res.ecl_spec / res.set_index(keys).index.map(b0.ecl_spec) - 1)
if use_segments:
    res["ecl_spec_lgd_reseg_pct_vs_base"] = 100 * (res.ecl_spec_lgd_reseg / res.set_index(keys).index.map(b0.ecl_spec) - 1)
res.to_csv(rep / f"stress_results{SUF}.csv", index=False)
pd.concat(pd_frames).to_parquet(art / "stress_pd.parquet", index=False)

cols = ["feature_set", "model", "scenario", "expected_defaults", "ecl_spec", "ecl_spec_pct_vs_base", "ecl_adj",
        "share_ltv_above_p99", "share_rate_above_p99"] + (["ecl_spec_lgd_reseg_pct_vs_base"] if use_segments else [])
show = res[res.population == "test"][cols]
print(f"\nTest population: {int(res[res.population == 'test'].loans.iloc[0]):,} loans, "
      f"{int(res[res.population == 'test'].observed_defaults.iloc[0])} observed defaults")
print(show.round(2).to_string(index=False))
print(f"\nWrote reports/lgd_segments.csv, reports/stress_results{SUF}.csv, artifacts/stress_pd.parquet")
