"""Split, tune, and evaluate the scorecard and the LightGBM challenger.

Usage:  py -m creditrisk.train [--features full|origination]
Reads:  data/processed/model_table.parquet, configs/project.yaml
Writes: artifacts/<set>/{split.parquet, scorecard.joblib, lgbm.joblib, predictions.parquet, run_summary.json}
        reports/<set>/{model_comparison.csv, calibration_table.csv, scorecard_coefficients.csv, sign_check.csv}

Feature sets: 'full' = origination fields plus as-of-prediction behaviour (delinquency history, balance ratio,
modification flag); 'origination' = underwriting fields only. <set> is the feature set name.

Design: one loan-level stratified split (development / test). Tuning and Platt recalibration use stratified
cross-validation inside the development set only. No resampling or class weights. Development-set
predictions in predictions.parquet are out-of-fold; test predictions come from models fitted on the full
development set. Test metrics carry bootstrap 95% intervals. All parameters come from configs/project.yaml.

Each model is one Pipeline over the raw loan table (see model.py), saved together with its Platt recalibration as a
CalibratedScorer, so scoring new loans applies exactly the transformations used in training. The modeling table is
validated (validation.py) before anything is fitted.
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from creditrisk.config import (
    FEATURE_SETS,
    common_args,
    feature_lists,
    load_config,
    project_paths,
)
from creditrisk.features import category_levels
from creditrisk.metrics import (
    apply_platt,
    bootstrap_indices,
    bootstrap_table,
    calibration_bins,
    fit_platt,
)
from creditrisk.model import CalibratedScorer, build_lgbm_pipeline, build_scorecard_pipeline
from creditrisk.validation import validate_model_table


def make_split(df, cfg):
    """Loan-level stratified split; returns (dev positions, test positions)."""
    return train_test_split(np.arange(len(df)), test_size=cfg["split"]["test_size"],
                            stratify=df.target.to_numpy(), random_state=cfg["seed"])


def fit_scorecard(dev, y, spec, cfg, cv):
    pipe = build_scorecard_pipeline(spec, cfg)
    search = GridSearchCV(pipe, {"lr__C": cfg["scorecard"]["c_grid"]}, scoring="neg_log_loss", cv=cv, n_jobs=-1)
    return search.fit(dev, y)


def fit_lgbm(dev, y, spec, cfg, levels, cv):
    grid = {f"model__{k}": v for k, v in cfg["lgbm"]["grid"].items()}
    search = GridSearchCV(build_lgbm_pipeline(spec, cfg, levels), grid, scoring="neg_log_loss", cv=cv, n_jobs=-1)
    return search.fit(dev, y)


def sign_check(coef_table, expected_signs):
    """Spearman correlation between bin order and coefficient for each feature with a stated expected sign."""
    rows = []
    for f, sgn in expected_signs.items():
        sub = coef_table[coef_table.feature.str.match(rf"^num__{f}__bin\d+$")]
        if len(sub) < 2:
            continue
        k = sub.feature.str.extract(r"bin(\d+)$")[0].astype(int)
        rho = spearmanr(k, sub.coefficient)[0]
        rows.append({"feature": f, "expected": "+" if sgn > 0 else "-", "n_bins": len(sub),
                     "spearman_bin_vs_coef": rho, "status": "ok" if np.sign(rho) == sgn else "CHECK"})
    return pd.DataFrame(rows)


def run(cfg, feature_set, log=print):
    paths = project_paths(cfg)
    art, rep = paths["artifacts"] / feature_set, paths["reports"] / feature_set
    art.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    seed, ev = cfg["seed"], cfg["evaluation"]
    eps, platt_c = ev["prob_clip"], ev["platt_c"]

    df = pd.read_parquet(paths["processed"] / "model_table.parquet")
    profile_table, _ = validate_model_table(df, cfg)
    paths["reports"].mkdir(parents=True, exist_ok=True)
    profile_table.to_csv(paths["reports"] / "validation_model_table.csv", index=False)
    spec = feature_lists(cfg, feature_set)
    log(f"feature set: {feature_set} - {len(spec['numeric']) + len(spec['categorical'])} features")

    dev_idx, test_idx = make_split(df, cfg)
    split = pd.DataFrame({"loan_id": df.loan_id, "split": "dev"})
    split.loc[test_idx, "split"] = "test"
    split.to_parquet(art / "split.parquet", index=False)
    dev, test = df.iloc[dev_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    y_dev, y_test = dev.target.to_numpy(), test.target.to_numpy()
    log(f"dev {len(dev):,} loans / {y_dev.sum()} defaults   test {len(test):,} loans / {y_test.sum()} defaults")

    cv = StratifiedKFold(cfg["split"]["cv_folds"], shuffle=True, random_state=seed)
    levels = category_levels(df, spec)

    sc = fit_scorecard(dev, y_dev, spec, cfg, cv)
    log(f"scorecard best: {sc.best_params_}")
    lgbm = fit_lgbm(dev, y_dev, spec, cfg, levels, cv)
    lgbm_params = {k.removeprefix("model__"): v for k, v in lgbm.best_params_.items()}
    log(f"lgbm best: {lgbm_params}")

    models = {"scorecard": sc.best_estimator_, "lgbm": lgbm.best_estimator_}
    raw_dev, raw_test, platt = {}, {}, {}
    for name, est in models.items():
        raw_dev[name] = cross_val_predict(est, dev, y_dev, cv=cv, method="predict_proba")[:, 1]
        raw_test[name] = est.predict_proba(test)[:, 1]
        platt[name] = fit_platt(raw_dev[name], y_dev, eps, platt_c)

    variants = {}
    for name in models:
        variants[f"{name}_raw"] = (raw_dev[name], raw_test[name])
        variants[f"{name}_platt"] = (apply_platt(raw_dev[name], platt[name], eps),
                                     apply_platt(raw_test[name], platt[name], eps))

    rng = np.random.default_rng(seed)
    boot_idx = bootstrap_indices(len(y_test), ev["n_bootstrap"], rng)
    rows, calib_rows, summary = [], [], {}
    for vname, (p_dev, p_test) in variants.items():
        thr = np.quantile(p_dev, 1 - ev["top_share"])
        est, vrows = bootstrap_table(vname, y_test, p_test, thr, boot_idx, eps, platt_c)
        rows += vrows
        calib_rows.append(calibration_bins(vname, y_test, p_test, ev["calibration_bins"]))
        summary[vname] = {"threshold_top10_from_dev_oof": float(thr), **{k: float(v) for k, v in est.items()}}
    pd.DataFrame(rows).to_csv(rep / "model_comparison.csv", index=False)
    pd.concat(calib_rows).to_csv(rep / "calibration_table.csv", index=False)

    mask = ~((y_test == 1) & (test.meta_disaster_event.to_numpy() == 1))
    sens = {n: {"roc_auc": float(roc_auc_score(y_test[mask], raw_test[n][mask])),
                "pr_auc": float(average_precision_score(y_test[mask], raw_test[n][mask]))} for n in models}

    pipe = sc.best_estimator_
    ctab = pd.DataFrame({"feature": pipe.named_steps["pre"].get_feature_names_out(),
                         "coefficient": pipe.named_steps["lr"].coef_[0]})
    ctab.to_csv(rep / "scorecard_coefficients.csv", index=False)
    signs = sign_check(ctab, cfg["features"]["expected_signs"])
    signs.to_csv(rep / "sign_check.csv", index=False)

    pred = pd.DataFrame({"loan_id": pd.concat([dev.loan_id, test.loan_id], ignore_index=True),
                         "split": ["dev"] * len(dev) + ["test"] * len(test),
                         "target": np.concatenate([y_dev, y_test])})
    for vname, (p_dev, p_test) in variants.items():
        pred[f"p_{vname}"] = np.concatenate([p_dev, p_test])
    pred.to_parquet(art / "predictions.parquet", index=False)

    for name, est in models.items():
        joblib.dump(CalibratedScorer(est, platt[name], eps), art / f"{name}.joblib")
    with open(art / "run_summary.json", "w") as fh:
        json.dump({"feature_set": feature_set, "seed": seed, "n_dev": len(dev), "n_test": len(test),
                   "events_dev": int(y_dev.sum()), "events_test": int(y_test.sum()),
                   "scorecard_params": sc.best_params_, "lgbm_params": lgbm_params, "platt": platt,
                   "test_metrics": summary, "sensitivity_excl_disaster_defaults": sens}, fh, indent=2, default=float)

    show = pd.DataFrame(rows).pivot(index="model", columns="metric", values="estimate")
    log(show[["roc_auc", "pr_auc", "brier", "cal_intercept", "cal_slope", "recall_top10"]].round(4).to_string())
    log("\nsign check:\n" + signs.round(2).to_string(index=False))
    log("\nsensitivity excluding disaster-flagged test defaults: " + json.dumps(sens))
    return {"summary": summary, "sign_check": signs, "sensitivity": sens}


def main(argv=None):
    ap = common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--features", choices=FEATURE_SETS, default="full")
    args = ap.parse_args(argv)
    run(load_config(args.config), args.features)


if __name__ == "__main__":
    main()
