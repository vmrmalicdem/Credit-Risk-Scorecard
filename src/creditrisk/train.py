"""Split, tune, and evaluate the scorecard and the LightGBM challenger.

Usage:  py src/creditrisk/train.py [--features full|origination]
Reads:  data/processed/model_table.parquet, data/processed/feature_spec.json, configs/project.yaml
Writes: artifacts/<set>/{split.parquet, scorecard.joblib, lgbm.joblib, predictions.parquet, run_summary.json}
        reports/<set>/{model_comparison.csv, calibration_table.csv, scorecard_coefficients.csv, sign_check.csv}

Feature sets: 'full' = origination fields plus as-of-prediction behaviour (delinquency history, balance ratio,
modification flag); 'origination' = underwriting fields only. <set> is the feature set name.

Design: one loan-level stratified split (development / test). Tuning and Platt recalibration use stratified
cross-validation inside the development set only. No resampling or class weights. Development-set
predictions in predictions.parquet are out-of-fold; test predictions come from models fitted on the full
development set. Test metrics carry bootstrap 95% intervals.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import QuantileBinner, category_levels, lgbm_frame, scorecard_frame

cfg = yaml.safe_load(open("configs/project.yaml"))
ap = argparse.ArgumentParser()
ap.add_argument("--features", choices=["full", "origination"], default="full")
FS = ap.parse_args().features
proc, art, rep = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["artifacts"]) / FS, Path("reports") / FS
art.mkdir(parents=True, exist_ok=True)
rep.mkdir(parents=True, exist_ok=True)
SEED, FOLDS, N_BOOT, TOP_SHARE = cfg["seed"], cfg["split"]["cv_folds"], 500, 0.10

df = pd.read_parquet(proc / "model_table.parquet")
spec = json.load(open(proc / "feature_spec.json"))
PRED = {"upb_ratio", "delinq_at_pred", "max_delinq_prior12", "ever_90_before", "mod_at_pred"}
if FS == "origination":
    spec = {k: [c for c in v if c not in PRED] for k, v in spec.items()}
print("feature set:", FS, "-", len(spec["numeric"]) + len(spec["categorical"]), "features")
y_all = df.target.to_numpy()

dev_idx, test_idx = train_test_split(np.arange(len(df)), test_size=cfg["split"]["test_size"],
                                     stratify=y_all, random_state=SEED)
split = pd.DataFrame({"loan_id": df.loan_id, "split": "dev"})
split.loc[test_idx, "split"] = "test"
split.to_parquet(art / "split.parquet", index=False)
dev, test = df.iloc[dev_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
y_dev, y_test = dev.target.to_numpy(), test.target.to_numpy()
print(f"dev {len(dev):,} loans / {y_dev.sum()} defaults   test {len(test):,} loans / {y_test.sum()} defaults")

cv = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
levels = category_levels(df, spec)

Xs_dev, Xs_test = scorecard_frame(dev, spec), scorecard_frame(test, spec)
pre = ColumnTransformer([
    ("num", QuantileBinner(5), spec["numeric"]),
    ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01), spec["categorical"]),
])
sc = GridSearchCV(Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=3000))]),
                  {"lr__C": [0.01, 0.03, 0.1, 0.3, 1.0]}, scoring="neg_log_loss", cv=cv, n_jobs=-1).fit(Xs_dev, y_dev)
print("scorecard best:", sc.best_params_)

Xg_dev, Xg_test = lgbm_frame(dev, spec, levels), lgbm_frame(test, spec, levels)
lgbm = GridSearchCV(
    LGBMClassifier(n_estimators=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                   n_jobs=1, verbose=-1, deterministic=True, force_row_wise=True),
    {"learning_rate": [0.02, 0.05], "num_leaves": [4, 8, 16], "min_child_samples": [50, 150], "reg_lambda": [1, 10]},
    scoring="neg_log_loss", cv=cv, n_jobs=-1).fit(Xg_dev, y_dev)
print("lgbm best:", lgbm.best_params_)

models = {"scorecard": (sc.best_estimator_, Xs_dev, Xs_test), "lgbm": (lgbm.best_estimator_, Xg_dev, Xg_test)}
clip = lambda p: np.clip(p, 1e-6, 1 - 1e-6)
raw_dev, raw_test, platt = {}, {}, {}
for name, (est, Xd, Xt) in models.items():
    raw_dev[name] = cross_val_predict(est, Xd, y_dev, cv=cv, method="predict_proba")[:, 1]
    raw_test[name] = est.predict_proba(Xt)[:, 1]
    pl = LogisticRegression(C=1e6).fit(logit(clip(raw_dev[name])).reshape(-1, 1), y_dev)
    platt[name] = (float(pl.intercept_[0]), float(pl.coef_[0, 0]))


def apply_platt(p, ab):
    return expit(ab[0] + ab[1] * logit(clip(p)))


variants = {}
for name in models:
    variants[f"{name}_raw"] = (raw_dev[name], raw_test[name])
    variants[f"{name}_platt"] = (apply_platt(raw_dev[name], platt[name]), apply_platt(raw_test[name], platt[name]))


def point_metrics(y, p, thr):
    z = logit(clip(p)).reshape(-1, 1)
    cal = LogisticRegression(C=1e6).fit(z, y)
    flag = p >= thr
    return {
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
        "cal_intercept": cal.intercept_[0], "cal_slope": cal.coef_[0, 0],
        "recall_top10": y[flag].sum() / y.sum(), "precision_top10": y[flag].mean() if flag.any() else np.nan,
        "mean_pred": p.mean(), "base_rate": y.mean(),
    }


rng = np.random.default_rng(SEED)
boot_idx = [rng.integers(0, len(y_test), len(y_test)) for _ in range(N_BOOT)]
rows, calib_rows, summary = [], [], {}
for vname, (p_dev, p_test) in variants.items():
    thr = np.quantile(p_dev, 1 - TOP_SHARE)
    est = point_metrics(y_test, p_test, thr)
    boots = pd.DataFrame([point_metrics(y_test[i], p_test[i], thr) for i in boot_idx])
    for m, v in est.items():
        rows.append({"model": vname, "metric": m, "estimate": v,
                     "ci_low": boots[m].quantile(0.025), "ci_high": boots[m].quantile(0.975)})
    q = pd.qcut(pd.Series(p_test).rank(method="first"), 5, labels=False)
    t = pd.DataFrame({"q": q, "p": p_test, "y": y_test}).groupby("q").agg(
        loans=("y", "size"), events=("y", "sum"), mean_pred=("p", "mean"), obs_rate=("y", "mean")).reset_index()
    t.insert(0, "model", vname)
    calib_rows.append(t)
    summary[vname] = {"threshold_top10_from_dev_oof": float(thr), **{k: float(v) for k, v in est.items()}}
pd.DataFrame(rows).to_csv(rep / "model_comparison.csv", index=False)
pd.concat(calib_rows).to_csv(rep / "calibration_table.csv", index=False)

mask = ~((y_test == 1) & (test.meta_disaster_event.to_numpy() == 1))
sens = {n: {"roc_auc": float(roc_auc_score(y_test[mask], raw_test[n][mask])),
            "pr_auc": float(average_precision_score(y_test[mask], raw_test[n][mask]))} for n in models}

names = sc.best_estimator_.named_steps["pre"].get_feature_names_out()
coefs = sc.best_estimator_.named_steps["lr"].coef_[0]
ctab = pd.DataFrame({"feature": names, "coefficient": coefs})
ctab.to_csv(rep / "scorecard_coefficients.csv", index=False)
expected = {"credit_score": -1, "ltv": 1, "cltv": 1, "dti": 1, "orig_rate": 1,
            "delinq_at_pred": 1, "max_delinq_prior12": 1, "ever_90_before": 1}
sign_rows = []
for f, sgn in expected.items():
    sub = ctab[ctab.feature.str.match(rf"^num__{f}__bin\d+$")]
    if len(sub) < 2:
        continue
    k = sub.feature.str.extract(r"bin(\d+)$")[0].astype(int)
    rho = spearmanr(k, sub.coefficient)[0]
    sign_rows.append({"feature": f, "expected": "+" if sgn > 0 else "-", "n_bins": len(sub),
                      "spearman_bin_vs_coef": rho, "status": "ok" if np.sign(rho) == sgn else "CHECK"})
pd.DataFrame(sign_rows).to_csv(rep / "sign_check.csv", index=False)

pred = pd.DataFrame({"loan_id": pd.concat([dev.loan_id, test.loan_id], ignore_index=True),
                     "split": ["dev"] * len(dev) + ["test"] * len(test),
                     "target": np.concatenate([y_dev, y_test])})
for vname, (p_dev, p_test) in variants.items():
    pred[f"p_{vname}"] = np.concatenate([p_dev, p_test])
pred.to_parquet(art / "predictions.parquet", index=False)

joblib.dump({"model": sc.best_estimator_, "platt": platt["scorecard"], "spec": spec}, art / "scorecard.joblib")
joblib.dump({"model": lgbm.best_estimator_, "platt": platt["lgbm"], "levels": levels, "spec": spec}, art / "lgbm.joblib")
json.dump({"feature_set": FS, "seed": SEED, "n_dev": len(dev), "n_test": len(test), "events_dev": int(y_dev.sum()),
           "events_test": int(y_test.sum()), "scorecard_params": sc.best_params_, "lgbm_params": lgbm.best_params_,
           "platt": platt, "test_metrics": summary, "sensitivity_excl_disaster_defaults": sens},
          open(art / "run_summary.json", "w"), indent=2, default=float)

show = pd.DataFrame(rows).pivot(index="model", columns="metric", values="estimate")
print(show[["roc_auc", "pr_auc", "brier", "cal_intercept", "cal_slope", "recall_top10"]].round(4).to_string())
print("\nsign check:\n", pd.DataFrame(sign_rows).round(2).to_string(index=False))
print("\nsensitivity excluding disaster-flagged test defaults:", json.dumps(sens))
