"""Final figures and markdown tables for the README.

Usage:  py src/creditrisk/report.py
Reads:  reports/{full,origination}/{model_comparison,calibration_table}.csv, reports/{lgd_segments,stress_results,data_funnel}.csv,
        artifacts/stress_pd.parquet, artifacts/<set>/split.parquet
Writes: reports/figures/fig1_discrimination.png, fig2_calibration.png, fig3_lgd.png, fig4_stress_ecl.png, fig5_pd_ratio.png
        reports/summary_tables.md

Figures follow one palette (blue = scorecard, orange = LightGBM) and show uncertainty where it exists. Calibration
bins are quintiles of predicted PD on the test set, so the low bins contain few or no defaults.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rep, art = Path("reports"), Path("artifacts")
fig_dir = rep / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
KINDS = ["scorecard", "lgbm"]
COL = {"scorecard": BLUE, "lgbm": ORANGE}
LAB = {"scorecard": "Scorecard (logistic)", "lgbm": "LightGBM challenger"}
FS = {"full": "Full: origination + behaviour", "origination": "Origination only"}
SCEN = ["baseline", "ltv_10", "ltv_20", "rate_200", "combined"]
SCEN_LAB = {"baseline": "Baseline", "ltv_10": "LTV +10 pp", "ltv_20": "LTV +20 pp",
            "rate_200": "Rate +200 bp", "combined": "LTV +20 pp\n+ rate +200 bp"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})

cmp = {fs: pd.read_csv(rep / fs / "model_comparison.csv") for fs in FS}
cal = {fs: pd.read_csv(rep / fs / "calibration_table.csv") for fs in FS}
lgd = pd.read_csv(rep / "lgd_segments.csv")
stress = pd.read_csv(rep / "stress_results.csv")
stress = stress[stress.population == "test"]
funnel = pd.read_csv(rep / "data_funnel.csv")


def metric(fs, variant, m):
    r = cmp[fs][(cmp[fs].model == variant) & (cmp[fs].metric == m)].iloc[0]
    return r.estimate, r.ci_low, r.ci_high


def save(fig, name):
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=200)
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
for ax, (m, title) in zip(axes, [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]):
    for j, kind in enumerate(KINDS):
        for i, fs in enumerate(FS):
            e, lo, hi = metric(fs, f"{kind}_raw", m)
            ax.errorbar(i + (j - 0.5) * 0.2, e, yerr=[[e - lo], [hi - e]], fmt="o", color=COL[kind], ms=8, lw=2,
                        capsize=4, mec=SURF, mew=1.5, label=LAB[kind] if i == 0 else None)
    ax.set_xticks(range(len(FS)))
    ax.set_xticklabels(list(FS.values()), fontsize=9)
    ax.set_xlim(-0.6, len(FS) - 0.4)
    ax.set_title(title, loc="left", fontsize=11)
    if m == "pr_auc":
        base = metric("full", "scorecard_raw", "base_rate")[0]
        ax.axhline(base, color=MUTED, ls="--", lw=1)
        ax.text(len(FS) - 0.45, base, f"base rate {base:.2%}", ha="right", va="bottom", fontsize=8, color=INK2)
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("Test-set discrimination, bootstrap 95% intervals", x=0.01, ha="left", fontsize=12)
save(fig, "fig1_discrimination.png")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
for ax, fs in zip(axes, FS):
    top = 0
    for kind in KINDS:
        d = cal[fs][cal[fs].model == f"{kind}_platt"].sort_values("q")
        ax.plot(d.mean_pred * 100, d.obs_rate * 100, "-o", color=COL[kind], lw=2, ms=6, mec=SURF, label=LAB[kind])
        top = max(top, d.mean_pred.max(), d.obs_rate.max())
    lim = top * 108
    ax.plot([0, lim], [0, lim], color=MUTED, ls="--", lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title(FS[fs], loc="left", fontsize=11)
    ax.set_xlabel("Mean predicted PD in bin, %")
    ax.set_ylabel("Observed default rate in bin, %")
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("Calibration on the test set (quintiles of predicted PD, Platt-calibrated)", x=0.01, ha="left", fontsize=12)
save(fig, "fig2_calibration.png")

fig, ax = plt.subplots(figsize=(8, 3.6))
order = ["all", "ltv_le_80", "ltv_gt_80"]
names = {"all": "Pooled", "ltv_le_80": "LTV at origination <= 80", "ltv_gt_80": "LTV at origination > 80"}
for i, seg in enumerate(order):
    r = lgd[lgd.segment == seg].iloc[0]
    ax.errorbar(r.mean_lgd * 100, i, xerr=[[(r.mean_lgd - r.ci_low) * 100], [(r.ci_high - r.mean_lgd) * 100]],
                fmt="o", color=BLUE if r.used_in_ecl else MUTED, ms=8, lw=2, capsize=4, mec=SURF, mew=1.5)
    tag = "  (used in ECL)" if r.used_in_ecl else ""
    ax.text(r.ci_high * 100 + 1.5, i, f"{r.mean_lgd:.1%}, n={int(r.n)}{tag}", va="center", fontsize=9, color=INK2)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([names[s] for s in order], fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, lgd.ci_high.max() * 100 + 25)
ax.set_xlabel("Mean realized LGD, % of balance at removal (bootstrap 95% interval)")
ax.set_title("Realized LGD on liquidated loans, floored at 0 and capped at 1", loc="left", fontsize=11)
save(fig, "fig3_lgd.png")

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True)
for ax, fs in zip(axes, FS):
    for j, kind in enumerate(KINDS):
        d = stress[(stress.feature_set == fs) & (stress.model == kind)].set_index("scenario").loc[SCEN]
        x = np.arange(len(SCEN)) + (j - 0.5) * 0.37
        bars = ax.bar(x, d.ecl_spec / 1e6, width=0.35, color=COL[kind], label=LAB[kind])
        for b, sc, pct in zip(bars, SCEN, d.ecl_spec_pct_vs_base):
            if sc in ("rate_200", "combined"):
                b.set_hatch("///")
                b.set_edgecolor(SURF)
            if sc != "baseline":
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{pct:+.0f}%", ha="center", va="bottom",
                        fontsize=8, color=INK2)
    ax.set_xticks(range(len(SCEN)))
    ax.set_xticklabels([SCEN_LAB[s] for s in SCEN], fontsize=8)
    ax.set_title(FS[fs], loc="left", fontsize=11)
axes[0].set_ylabel("Simplified 12-month ECL, test portfolio (USD millions)")
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("Baseline vs stressed ECL (PD x pooled LGD x EAD). Hatched = rate scenarios, extrapolated beyond observed rates",
             x=0.01, ha="left", fontsize=11)
save(fig, "fig4_stress_ecl.png")

pdv = pd.read_parquet(art / "stress_pd.parquet")
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
colors = {"ltv_10": BLUE, "ltv_20": ORANGE, "rate_200": AQUA, "combined": YELLOW}
for ax, fs in zip(axes, FS):
    test_ids = set(pd.read_parquet(art / fs / "split.parquet").query("split == 'test'").loan_id)
    d = pdv[(pdv.feature_set == fs) & (pdv.model == "scorecard") & pdv.loan_id.isin(test_ids)]
    w = d.pivot(index="loan_id", columns="scenario", values="pd")
    for sc, c in colors.items():
        r = np.sort((w[sc] / w["baseline"]).to_numpy())
        ax.plot(r, np.arange(1, len(r) + 1) / len(r), color=c, lw=2, label=SCEN_LAB[sc].replace("\n", " "))
    ax.axvline(1, color=MUTED, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_title(FS[fs], loc="left", fontsize=11)
    ax.set_xlabel("PD ratio, stressed / baseline (log scale)")
axes[0].set_ylabel("Share of test loans")
axes[1].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("Loan-level PD response to each shock, scorecard", x=0.01, ha="left", fontsize=12)
save(fig, "fig5_pd_ratio.png")


def fmt(t, nd):
    return f"{t[0]:.{nd}f} ({t[1]:.{nd}f} to {t[2]:.{nd}f})"


def md(df):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.to_numpy()]
    return "\n".join(lines)


rows = []
for fs in FS:
    for kind in KINDS:
        rows.append({
            "Feature set": FS[fs], "Model": LAB[kind],
            "ROC-AUC": fmt(metric(fs, f"{kind}_raw", "roc_auc"), 3),
            "PR-AUC": fmt(metric(fs, f"{kind}_raw", "pr_auc"), 3),
            "Recall in top 10% of scores": fmt(metric(fs, f"{kind}_raw", "recall_top10"), 2),
            "Brier (Platt)": fmt(metric(fs, f"{kind}_platt", "brier"), 4),
            "Calibration slope (Platt)": fmt(metric(fs, f"{kind}_platt", "cal_slope"), 2),
        })
base = metric("full", "scorecard_raw", "base_rate")[0]
t_models = md(pd.DataFrame(rows))

t_lgd = md(pd.DataFrame({
    "Segment": lgd.segment, "Used in ECL": lgd.used_in_ecl, "n": lgd.n,
    "Mean LGD (95% interval)": [f"{r.mean_lgd:.3f} ({r.ci_low:.3f} to {r.ci_high:.3f})" for r in lgd.itertuples()],
    "Median": lgd.median_lgd.round(3), "Balance-weighted mean": lgd.mean_lgd_balance_weighted.round(3),
    "Raw mean (unclipped)": lgd.mean_raw_unclipped.round(3), "Floored": lgd.n_floored, "Capped": lgd.n_capped,
}))

t_stress = md(pd.DataFrame({
    "Feature set": stress.feature_set, "Model": stress.model, "Scenario": stress.scenario,
    "Expected defaults": stress.expected_defaults.round(1),
    "ECL spec (USD M)": (stress.ecl_spec / 1e6).round(3), "Change vs baseline %": stress.ecl_spec_pct_vs_base.round(1),
    "ECL adj (USD M)": (stress.ecl_adj / 1e6).round(3),
    "Loans beyond p99 of observed LTV": stress.share_ltv_above_p99.round(2),
    "Loans beyond p99 of observed rate": stress.share_rate_above_p99.round(2),
}))

out = [
    "# Summary tables (generated by report.py)\n",
    "## Data funnel\n", md(funnel.rename(columns={"step": "Step", "loans": "Loans"})), "",
    "## Model comparison (test set, bootstrap 95% intervals)\n", t_models,
    f"\nTest base rate {base:.4%}; Brier score of a constant prediction at the base rate is {base * (1 - base):.4f}.\n",
    "## Realized LGD (liquidated loans)\n", t_lgd, "",
    f"## Baseline vs stressed ECL (test portfolio, {int(stress.loans.iloc[0]):,} loans, "
    f"{int(stress.observed_defaults.iloc[0])} observed defaults)\n", t_stress, "",
]
(rep / "summary_tables.md").write_text("\n".join(out))
print("Wrote 5 figures to reports/figures/ and reports/summary_tables.md")
