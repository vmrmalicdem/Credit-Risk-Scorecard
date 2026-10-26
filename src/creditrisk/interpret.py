"""Scorecard points table and LightGBM TreeSHAP importance.

Usage:  py src/creditrisk/interpret.py
Reads:  data/processed/model_table.parquet, artifacts/<set>/{scorecard.joblib, lgbm.joblib, split.parquet}
Writes: reports/<set>/scorecard_table.csv, reports/<set>/lgbm_shap_importance.csv, reports/scorecard_points.md,
        reports/figures/fig6_scorecard_points.png, reports/figures/fig7_lgbm_shap.png

Points scaling: score = offset + factor x ln(good:bad odds), base score 600 at 200:1 odds (about 0.5% PD) and
20 points to double the odds; higher score = lower risk. Bin points = -factor x coefficient and
base points = offset - factor x intercept, so a loan's score is the base points plus the points of its bins.
Points describe the raw logistic model, before Platt recalibration. Bin counts and default rates are development-set values.
SHAP values are LightGBM's built-in TreeSHAP contributions (log-odds) on the test set.
"""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from features import lgbm_frame, scorecard_frame

cfg = yaml.safe_load(open("configs/project.yaml"))
proc, art, rep = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["artifacts"]), Path("reports")
fig_dir = rep / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

PDO, BASE_SCORE, BASE_ODDS, MIN_N, TOP = 20, 600, 200, 50, 12
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)
FS = {"full": "Full: origination + behaviour", "origination": "Origination only"}

SURF, INK, INK2, MUTED, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})

df = pd.read_parquet(proc / "model_table.parquet")


def numeric_labels(binner):
    out = {}
    for c in binner.columns_:
        e = binner.edges_[c]
        labs = [f"<= {e[0]:g}"] if len(e) else ["all"]
        labs += [f"> {e[k - 1]:g} to {e[k]:g}" for k in range(1, len(e))]
        if len(e):
            labs.append(f"> {e[-1]:g}")
        if binner.has_na_[c]:
            labs.append("missing")
        out[c] = labs
    return out


def feature_bins(pre, spec):
    labs = numeric_labels(pre.named_transformers_["num"])
    cats = sorted(spec["categorical"], key=len, reverse=True)
    rows = []
    for name in pre.get_feature_names_out():
        kind, rest = name.split("__", 1)
        if kind == "num":
            feat, tail = rest.rsplit("__", 1)
            k = len(labs[feat]) - 1 if tail == "missing" else int(tail.replace("bin", ""))
            rows.append((feat, labs[feat][k]))
        else:
            feat = next(c for c in cats if rest.startswith(c + "_"))
            level = rest[len(feat) + 1:]
            rows.append((feat, "other (rare levels)" if level == "infrequent_sklearn" else level))
    return rows


def md(df_):
    lines = ["| " + " | ".join(df_.columns) + " |", "|" + "---|" * len(df_.columns)]
    return "\n".join(lines + ["| " + " | ".join(map(str, r)) + " |" for r in df_.to_numpy()])


tables, ranges, shap_tabs = {}, {}, {}
for fs in FS:
    split = pd.read_parquet(art / fs / "split.parquet").set_index("loan_id").split
    dev = df[df.loan_id.map(split) == "dev"].reset_index(drop=True)
    test = df[df.loan_id.map(split) == "test"].reset_index(drop=True)

    sc = joblib.load(art / fs / "scorecard.joblib")
    spec, pipe = sc["spec"], sc["model"]
    pre, lr = pipe.named_steps["pre"], pipe.named_steps["lr"]
    M = pre.transform(scorecard_frame(dev, spec))
    y = dev.target.to_numpy()
    counts = np.asarray(M.sum(axis=0)).ravel()
    events = np.asarray(M.T.dot(y)).ravel()
    bins = feature_bins(pre, spec)
    t = pd.DataFrame({"feature": [b[0] for b in bins], "bin": [b[1] for b in bins],
                      "coefficient": lr.coef_[0], "points": -FACTOR * lr.coef_[0],
                      "loans_dev": counts.astype(int), "defaults_dev": events.astype(int)})
    t["default_rate_dev"] = t.defaults_dev / t.loans_dev.where(t.loans_dev > 0)
    base = pd.DataFrame([{"feature": "BASE", "bin": "intercept", "coefficient": lr.intercept_[0],
                          "points": OFFSET - FACTOR * lr.intercept_[0], "loans_dev": len(dev),
                          "defaults_dev": int(y.sum()), "default_rate_dev": y.mean()}])
    t = pd.concat([base, t], ignore_index=True)
    t.to_csv(rep / fs / "scorecard_table.csv", index=False)
    tables[fs] = t
    body = t[(t.feature != "BASE") & (t.loans_dev >= MIN_N)]
    ranges[fs] = (body.groupby("feature").points.agg(lambda s: s.max() - s.min())
                  .sort_values(ascending=False))

    lg = joblib.load(art / fs / "lgbm.joblib")
    X = lgbm_frame(test, lg["spec"], lg["levels"])
    contrib = lg["model"].booster_.predict(X, pred_contrib=True)
    assert contrib.shape[1] == X.shape[1] + 1
    s = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(contrib[:, :-1]).mean(axis=0)})
    s["share_of_total"] = s.mean_abs_shap / s.mean_abs_shap.sum()
    s = s.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    s.to_csv(rep / fs / "lgbm_shap_importance.csv", index=False)
    shap_tabs[fs] = s


def barpanel(data, color, title, xlabel, fname, fmt):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, fs in zip(axes, FS):
        d = data[fs].head(TOP)[::-1]
        ax.barh(d.index if isinstance(d, pd.Series) else d.feature, d.values if isinstance(d, pd.Series) else d.mean_abs_shap,
                color=color, height=0.65)
        vals = d.values if isinstance(d, pd.Series) else d.mean_abs_shap.values
        for i, v in enumerate(vals):
            ax.text(v, i, " " + fmt(v), va="center", fontsize=8, color=INK2)
        ax.set_title(FS[fs], loc="left", fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, max(vals) * 1.18)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / fname, dpi=200)
    plt.close(fig)


barpanel(ranges, BLUE, "Scorecard: points range per feature (bins with at least 50 development loans)",
         "Points between the lowest- and highest-risk bin", "fig6_scorecard_points.png", lambda v: f"{v:.0f}")
barpanel(shap_tabs, ORANGE, "LightGBM challenger: mean |SHAP| on the test set (log-odds)",
         "Mean absolute contribution to log-odds of default", "fig7_lgbm_shap.png", lambda v: f"{v:.2f}")

lines = ["# Scorecard points (generated by interpret.py)\n",
         f"Base score {BASE_SCORE} at {BASE_ODDS}:1 good:bad odds, {PDO} points to double the odds. "
         "Score = base points + sum of bin points; higher = lower risk. Counts and default rates are development-set values.\n"]
for fs, t in tables.items():
    lines += [f"## {FS[fs]}\n", f"Base points: {t.points.iloc[0]:.0f}\n"]
    for f, g in t[t.feature != "BASE"].groupby("feature", sort=False):
        if len(g) > 12:
            lines.append(f"**{f}**: {len(g)} levels, see `reports/{fs}/scorecard_table.csv`.\n")
            continue
        g = g.assign(points=g.points.round(0).astype(int), default_rate_dev=g.default_rate_dev.map(lambda v: f"{v:.2%}"))
        lines += [f"**{f}**\n", md(g[["bin", "points", "loans_dev", "default_rate_dev"]]), ""]
(rep / "scorecard_points.md").write_text("\n".join(lines))

for fs in FS:
    print(f"\n{FS[fs]}")
    print("scorecard points range (top 8):", ranges[fs].head(8).round(0).astype(int).to_dict())
    print("LightGBM mean |SHAP| (top 8):", shap_tabs[fs].head(8).set_index("feature").mean_abs_shap.round(3).to_dict())
print("\nWrote scorecard tables, SHAP tables, reports/scorecard_points.md and figures 6-7")
