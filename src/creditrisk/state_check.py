"""Is `state` signal or noise? Defaults by state, and a refit without `state`.

Usage:  py -m creditrisk.state_check [--features full|origination]
Reads:  data/processed/model_table.parquet, artifacts/<set>/{split,predictions}.parquet,
        artifacts/<set>/{scorecard,lgbm}.joblib
Writes: reports/<set>/state_defaults.csv, reports/<set>/state_ablation.csv

Part 1: loans, defaults and default rate per state on the development set, with a Wilson 95% interval.
Part 2: refits both models without `state` on the same development loans (scorecard penalty re-tuned by CV, LightGBM
reusing the hyperparameters already chosen) and compares test ROC-AUC and PR-AUC with the saved models. The
differences carry paired bootstrap 95% intervals over the same test loans. An interval that contains 0 means the data
cannot show that `state` helps.
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from creditrisk.config import FEATURE_SETS, common_args, load_config, project_paths
from creditrisk.features import category_levels, lgbm_frame, scorecard_frame
from creditrisk.metrics import bootstrap_indices
from creditrisk.train import fit_scorecard


def state_defaults(dev):
    g = dev.groupby("state").target.agg(loans="size", defaults="sum").reset_index()
    z = 1.96
    p = g.defaults / g.loans
    den = 1 + z ** 2 / g.loans
    ctr = (p + z ** 2 / (2 * g.loans)) / den
    half = z * np.sqrt(p * (1 - p) / g.loans + z ** 2 / (4 * g.loans ** 2)) / den
    g["default_rate"], g["ci_low"], g["ci_high"] = p, ctr - half, ctr + half
    return g.sort_values("defaults", ascending=False)


def run(cfg, feature_set, log=print):
    paths = project_paths(cfg)
    art, rep = paths["artifacts"] / feature_set, paths["reports"] / feature_set
    seed = cfg["seed"]

    df = pd.read_parquet(paths["processed"] / "model_table.parquet")
    split = pd.read_parquet(art / "split.parquet").set_index("loan_id").split
    pred = pd.read_parquet(art / "predictions.parquet").set_index("loan_id")
    dev = df[df.loan_id.map(split) == "dev"].reset_index(drop=True)
    test = df[df.loan_id.map(split) == "test"].reset_index(drop=True)

    g = state_defaults(dev)
    g.to_csv(rep / "state_defaults.csv", index=False)
    log(f"development base rate {dev.target.mean():.2%}; {len(g)} states")
    log(f"states with fewer than 10 defaults: {(g.defaults < 10).sum()} of {len(g)}")
    log(g.head(10).round(4).to_string(index=False))

    lg_old = joblib.load(art / "lgbm.joblib")
    spec = {k: [c for c in v if c != "state"] for k, v in lg_old["spec"].items()}
    levels = category_levels(df, spec)
    y_dev, y_test = dev.target.to_numpy(), test.target.to_numpy()
    cv = StratifiedKFold(cfg["split"]["cv_folds"], shuffle=True, random_state=seed)

    sc = fit_scorecard(scorecard_frame(dev, spec), y_dev, spec, cfg, cv)
    lg = LGBMClassifier(**lg_old["model"].get_params()).fit(lgbm_frame(dev, spec, levels), y_dev)
    new = {"scorecard": sc.predict_proba(scorecard_frame(test, spec))[:, 1],
           "lgbm": lg.predict_proba(lgbm_frame(test, spec, levels))[:, 1]}
    old = {k: pred.loc[test.loan_id, f"p_{k}_raw"].to_numpy() for k in new}

    rng = np.random.default_rng(seed)
    idx = [i for i in bootstrap_indices(len(test), cfg["evaluation"]["n_bootstrap"], rng) if y_test[i].sum() > 0]
    rows = []
    for k in new:
        for name, fn in [("roc_auc", roc_auc_score), ("pr_auc", average_precision_score)]:
            a, b = fn(y_test, old[k]), fn(y_test, new[k])
            d = [fn(y_test[i], new[k][i]) - fn(y_test[i], old[k][i]) for i in idx]
            rows.append({"feature_set": feature_set, "model": k, "metric": name, "with_state": a,
                         "without_state": b, "diff": b - a,
                         "diff_ci_low": np.quantile(d, 0.025), "diff_ci_high": np.quantile(d, 0.975)})
    out = pd.DataFrame(rows)
    out.to_csv(rep / "state_ablation.csv", index=False)
    log("\nTest metrics with vs without state (diff = without minus with):")
    log(out.round(4).to_string(index=False))
    return out


def main(argv=None):
    ap = common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--features", choices=FEATURE_SETS, default="full")
    args = ap.parse_args(argv)
    run(load_config(args.config), args.features)


if __name__ == "__main__":
    main()
