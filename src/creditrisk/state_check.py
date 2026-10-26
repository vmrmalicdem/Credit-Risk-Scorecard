"""Is `state` signal or noise? Defaults by state, and a refit without `state`.

Usage:  py src/creditrisk/state_check.py [--features full|origination]
Reads:  data/processed/model_table.parquet, artifacts/<set>/{split,predictions}.parquet, artifacts/<set>/{scorecard,lgbm}.joblib
Writes: reports/<set>/state_defaults.csv, reports/<set>/state_ablation.csv

Part 1: loans, defaults and default rate per state on the development set, with a Wilson 95% interval.
Part 2: refits both models without `state` on the same development loans (scorecard penalty re-tuned by CV, LightGBM
reusing the hyperparameters already chosen) and compares test ROC-AUC and PR-AUC with the saved models. The
differences carry paired bootstrap 95% intervals over the same test loans. An interval that contains 0 means the data
cannot show that `state` helps.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import QuantileBinner, category_levels, lgbm_frame, scorecard_frame

ap = argparse.ArgumentParser()
ap.add_argument("--features", choices=["full", "origination"], default="full")
FS = ap.parse_args().features
cfg = yaml.safe_load(open("configs/project.yaml"))
proc, art, rep = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["artifacts"]) / FS, Path("reports") / FS
SEED, N_BOOT = cfg["seed"], 500

df = pd.read_parquet(proc / "model_table.parquet")
split = pd.read_parquet(art / "split.parquet").set_index("loan_id").split
pred = pd.read_parquet(art / "predictions.parquet").set_index("loan_id")
dev = df[df.loan_id.map(split) == "dev"].reset_index(drop=True)
test = df[df.loan_id.map(split) == "test"].reset_index(drop=True)

z = 1.96
g = dev.groupby("state").target.agg(loans="size", defaults="sum").reset_index()
p = g.defaults / g.loans
den = 1 + z ** 2 / g.loans
ctr = (p + z ** 2 / (2 * g.loans)) / den
half = z * np.sqrt(p * (1 - p) / g.loans + z ** 2 / (4 * g.loans ** 2)) / den
g["default_rate"], g["ci_low"], g["ci_high"] = p, ctr - half, ctr + half
g = g.sort_values("defaults", ascending=False)
g.to_csv(rep / "state_defaults.csv", index=False)
print(f"development base rate {dev.target.mean():.2%}; {len(g)} states")
print(f"states with fewer than 10 defaults: {(g.defaults < 10).sum()} of {len(g)}")
print(g.head(10).round(4).to_string(index=False))

sc_old, lg_old = joblib.load(art / "scorecard.joblib"), joblib.load(art / "lgbm.joblib")
spec = {k: [c for c in v if c != "state"] for k, v in sc_old["spec"].items()}
levels = category_levels(df, spec)
y_dev, y_test = dev.target.to_numpy(), test.target.to_numpy()
cv = StratifiedKFold(cfg["split"]["cv_folds"], shuffle=True, random_state=SEED)

pre = ColumnTransformer([
    ("num", QuantileBinner(5), spec["numeric"]),
    ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01), spec["categorical"]),
])
sc = GridSearchCV(Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=3000))]),
                  {"lr__C": [0.01, 0.03, 0.1, 0.3, 1.0]}, scoring="neg_log_loss", cv=cv, n_jobs=-1
                  ).fit(scorecard_frame(dev, spec), y_dev)
lg = LGBMClassifier(**lg_old["model"].get_params()).fit(lgbm_frame(dev, spec, levels), y_dev)

new = {"scorecard": sc.predict_proba(scorecard_frame(test, spec))[:, 1],
       "lgbm": lg.predict_proba(lgbm_frame(test, spec, levels))[:, 1]}
old = {k: pred.loc[test.loan_id, f"p_{k}_raw"].to_numpy() for k in new}

rng = np.random.default_rng(SEED)
idx = [rng.integers(0, len(test), len(test)) for _ in range(N_BOOT)]
idx = [i for i in idx if y_test[i].sum() > 0]
rows = []
for k in new:
    for name, fn in [("roc_auc", roc_auc_score), ("pr_auc", average_precision_score)]:
        a, b = fn(y_test, old[k]), fn(y_test, new[k])
        d = [fn(y_test[i], new[k][i]) - fn(y_test[i], old[k][i]) for i in idx]
        rows.append({"feature_set": FS, "model": k, "metric": name, "with_state": a, "without_state": b,
                     "diff": b - a, "diff_ci_low": np.quantile(d, 0.025), "diff_ci_high": np.quantile(d, 0.975)})
out = pd.DataFrame(rows)
out.to_csv(rep / "state_ablation.csv", index=False)
print("\nTest metrics with vs without state (diff = without minus with):")
print(out.round(4).to_string(index=False))
