"""LGD segments, simplified 12-month expected loss, and feature-shock stress test.

Usage:  py -m creditrisk.ecl_stress [--lgd pooled|segmented]
Reads:  data/processed/{model_table,liquidations}.parquet, artifacts/<set>/{scorecard,lgbm}.joblib,
        artifacts/<set>/{split,predictions}.parquet  (both feature sets from train must exist)
Writes: reports/lgd_segments.csv, reports/stress_results.csv, artifacts/stress_pd.parquet

LGD: realized loss / balance at removal for credit-event terminations with a reported actual loss, floored and
capped (configs/project.yaml, lgd). Default is one pooled average. With --lgd segmented, two segments by origination
LTV are used if each has at least lgd.min_segment_loans loans; segment results then carry a _segmented suffix.

ECL (USD, 12 months, simplified, not IFRS 9): per loan PD x LGD x EAD, summed over the portfolio.
  ecl_spec  uses the liquidation-only LGD and treats every default as a liquidation: a conservative upper bound.
  ecl_adj   multiplies by the observed share of in-window defaults that later ended in a liquidation with loss.
  ecl_spec_lgd_reseg  (segmented mode only) re-assigns the LGD segment from the shocked LTV; otherwise LGD is
  held at baseline.
PD is the Platt-calibrated probability. EAD is current balance at the prediction month.

Scenarios (cfg stress.scenarios) are sensitivity analyses of fitted models, not causal estimates or
macroeconomic forecasts. Baseline PDs on the test set must reproduce predictions.parquet.
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from creditrisk.config import common_args, load_config, project_paths
from creditrisk.ecl import (
    assign_lgd,
    expected_loss,
    lgd_table,
    prepare_liquidations,
    shock,
    wilson_interval,
)
from creditrisk.features import lgbm_frame, scorecard_frame
from creditrisk.metrics import apply_platt


def score(bundle, d, kind, eps):
    spec = bundle["spec"]
    X = scorecard_frame(d, spec) if kind == "scorecard" else lgbm_frame(d, spec, bundle["levels"])
    return apply_platt(bundle["model"].predict_proba(X)[:, 1], bundle["platt"], eps)


def run(cfg, lgd_mode="pooled", log=print):
    paths = project_paths(cfg)
    art, rep = paths["artifacts"], paths["reports"]
    rep.mkdir(exist_ok=True)
    lcfg, scfg = cfg["lgd"], cfg["stress"]
    eps = cfg["evaluation"]["prob_clip"]

    df = pd.read_parquet(paths["processed"] / "model_table.parquet")
    liq = prepare_liquidations(pd.read_parquet(paths["processed"] / "liquidations.parquet"),
                               lcfg["ltv_split"], lcfg["floor"], lcfg["cap"])
    rng = np.random.default_rng(cfg["seed"])
    lgd_tab, use_segments, lgd_map, lgd_pooled = lgd_table(
        liq, lgd_mode, lcfg["min_segment_loans"], rng, lcfg["bootstrap_resamples"])
    suffix = "_segmented" if use_segments else ""
    lgd_tab.to_csv(rep / "lgd_segments.csv", index=False)

    def lgd_for(ltv):
        return assign_lgd(ltv, use_segments, lgd_map, lgd_pooled, lcfg["ltv_split"])

    defaults = df.loc[df.target == 1, "loan_id"]
    conv_n, conv_d = int(defaults.isin(set(liq.loan_id)).sum()), len(defaults)
    conv = conv_n / conv_d
    lo, hi = wilson_interval(conv_n, conv_d)
    log(lgd_tab.round(3).to_string(index=False))
    log(f"LGD mode: {lgd_mode}; segments used: {use_segments}")
    log(f"in-window defaults that later liquidated with a reported loss: {conv_n}/{conv_d} = {conv:.1%} "
        f"(95% CI {lo:.1%} to {hi:.1%})")

    ltv_ref = float(df.ltv.quantile(scfg["reference_quantile"]))
    rate_ref = float(df.orig_rate.quantile(scfg["reference_quantile"]))
    lgd_base = lgd_for(df.ltv.to_numpy())
    ead_all = df.ead.to_numpy()
    rows, pd_frames = [], []
    for fs in ["full", "origination"]:
        split = pd.read_parquet(art / fs / "split.parquet").set_index("loan_id").split
        pred = pd.read_parquet(art / fs / "predictions.parquet").set_index("loan_id")
        is_test = df.loan_id.map(split).eq("test").to_numpy()
        for kind in ["scorecard", "lgbm"]:
            bundle = joblib.load(art / fs / f"{kind}.joblib")
            base = None
            for sname, sc in scfg["scenarios"].items():
                d = shock(df, sc["ltv_pp"], sc["rate_pts"], scfg["ltv_cap"])
                p = score(bundle, d, kind, eps)
                if sname == "baseline":
                    assert np.allclose(p[is_test], pred.loc[df.loan_id[is_test], f"p_{kind}_platt"].to_numpy(),
                                       atol=1e-6), f"baseline mismatch for {fs}/{kind}"
                    base = p
                lgd_shock = lgd_for(d.ltv.to_numpy())
                pd_frames.append(pd.DataFrame({"loan_id": df.loan_id, "feature_set": fs, "model": kind,
                                               "scenario": sname, "pd": p}))
                for popn, m in [("test", is_test), ("all", np.ones(len(df), bool))]:
                    ecl = expected_loss(p[m], lgd_base[m], ead_all[m])
                    row = {
                        "feature_set": fs, "model": kind, "population": popn, "scenario": sname,
                        "loans": int(m.sum()), "observed_defaults": int(df.target[m].sum()),
                        "expected_defaults": float(p[m].sum()), "mean_pd": float(p[m].mean()),
                        "mean_delta_pd": float((p[m] - base[m]).mean()),
                        "share_pd_up": float((p[m] > base[m]).mean()),
                        "ead_total": float(ead_all[m].sum()), "ecl_spec": ecl,
                        "ecl_spec_rate": ecl / float(ead_all[m].sum()), "ecl_adj": ecl * conv,
                        "share_ltv_above_p99": float((d.ltv[m] > ltv_ref).mean()),
                        "share_rate_above_p99": float((d.orig_rate[m] > rate_ref).mean()),
                    }
                    if use_segments:
                        row["ecl_spec_lgd_reseg"] = expected_loss(p[m], lgd_shock[m], ead_all[m])
                    rows.append(row)

    res = pd.DataFrame(rows)
    keys = ["feature_set", "model", "population"]
    b0 = res[res.scenario == "baseline"].set_index(keys)
    res["ecl_spec_pct_vs_base"] = 100 * (res.ecl_spec / res.set_index(keys).index.map(b0.ecl_spec) - 1)
    if use_segments:
        res["ecl_spec_lgd_reseg_pct_vs_base"] = 100 * (
            res.ecl_spec_lgd_reseg / res.set_index(keys).index.map(b0.ecl_spec) - 1)
    res.to_csv(rep / f"stress_results{suffix}.csv", index=False)
    pd.concat(pd_frames).to_parquet(art / "stress_pd.parquet", index=False)

    cols = ["feature_set", "model", "scenario", "expected_defaults", "ecl_spec", "ecl_spec_pct_vs_base", "ecl_adj",
            "share_ltv_above_p99", "share_rate_above_p99"]
    if use_segments:
        cols.append("ecl_spec_lgd_reseg_pct_vs_base")
    test_rows = res[res.population == "test"]
    log(f"\nTest population: {int(test_rows.loans.iloc[0]):,} loans, "
        f"{int(test_rows.observed_defaults.iloc[0])} observed defaults")
    log(test_rows[cols].round(2).to_string(index=False))
    log(f"\nWrote lgd_segments.csv, stress_results{suffix}.csv, stress_pd.parquet")
    return res


def main(argv=None):
    ap = common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--lgd", choices=["pooled", "segmented"], default="pooled")
    args = ap.parse_args(argv)
    run(load_config(args.config), args.lgd)


if __name__ == "__main__":
    main()
