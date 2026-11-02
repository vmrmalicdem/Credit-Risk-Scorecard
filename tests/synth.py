"""Tiny synthetic Freddie Mac-format files (pipe-delimited, no header) with hand-built loan scenarios.

Every scenario loan has first payment 2015-01, so the prediction month (loan age 24) is 2016-12 and the 12-month
outcome window is 2017-01 to 2017-12. Expected behaviour of each scenario is listed in SCENARIOS.
"""
import numpy as np
import pandas as pd

FP_IDX = 2015 * 12
PRED_IDX = FP_IDX + 23
LAST_IDX = PRED_IDX + 15


def period(idx):
    return f"{idx // 12}{idx % 12 + 1:02d}"


# scenario -> (eligible, target, note)
SCENARIOS = {
    "clean": (True, 0, "never late"),
    "default_in_window": (True, 1, "90+ days late 5 months after the prediction month"),
    "default_after_window": (True, 0, "90+ days late 13 months after the prediction month"),
    "late_at_pred": (False, None, "already 90+ days late at the prediction month"),
    "late30_then_default": (True, 1, "30 days late at prediction, 90+ three months later"),
    "payoff_in_window": (True, 0, "paid off (code 01) inside the window"),
    "excluded_16": (False, None, "terminated with code 16 inside the window"),
    "credit_event_09": (True, 1, "foreclosure alternative (09) inside the window, with a loss"),
    "terminated_before_pred": (False, None, "no performance row at the prediction month"),
    "first_payment_too_late": (False, None, "first payment after max_first_payment"),
    "cured_90_before": (True, 0, "90+ days late 5 months before prediction, cured"),
    "modified_at_pred": (True, 0, "modification flag set at the prediction month"),
}


def _row(loan_id, idx, k, status="00", mod="", zb="", loss="", removal="", disaster="", assist="", upb=None):
    return {"loan_id": loan_id, "period": period(idx),
            "current_upb": f"{(200000 - 400 * k) if upb is None else upb:.2f}", "delinq_status": status,
            "mod_flag": mod, "zero_bal_code": zb, "actual_loss": loss, "zero_bal_removal_upb": removal,
            "disaster_delinq_flag": disaster, "borrower_assist_plan": assist}


def _scenario_rows(name, loan_id, fp_idx):
    start = fp_idx
    rows = []
    last = start + 23 + 15
    for k in range(0, last - start + 1):
        idx = start + k
        status, kw = "00", {}
        off = idx - (start + 23)
        if name == "default_in_window" and off >= 5:
            status = "03"
        if name == "default_after_window" and off >= 13:
            status = "03"
        if name == "late_at_pred" and off >= 0:
            status = "03"
        if name == "late30_then_default":
            status = "01" if off == 0 else ("03" if off >= 3 else "00")
        if name == "cured_90_before" and -5 <= off <= -4:
            status = "03"
        if name == "modified_at_pred" and off >= 0:
            kw["mod"] = "Y"
        if name == "default_in_window" and off == 5:
            kw["disaster"] = "Y"
        if name == "payoff_in_window" and off == 6:
            rows.append(_row(loan_id, idx, k, zb="01", upb=0))
            break
        if name == "excluded_16" and off == 6:
            rows.append(_row(loan_id, idx, k, zb="16", upb=0))
            break
        if name == "credit_event_09" and off == 8:
            rows.append(_row(loan_id, idx, k, status="03", zb="09", loss="30000.00", removal="180000.00", upb=0))
            break
        if name == "terminated_before_pred" and off == -1:
            rows.append(_row(loan_id, idx, k, zb="01", upb=0))
            break
        rows.append(_row(loan_id, idx, k, status=status, **kw))
    return rows


def make_loans(n_clean=150, seed=0):
    rng = np.random.default_rng(seed)
    loans = [(f"S{i:02d}", name) for i, name in enumerate(SCENARIOS)]
    loans += [(f"C{i:04d}", "clean") for i in range(n_clean)]
    orig = []
    for loan_id, name in loans:
        fp = 201701 if name == "first_payment_too_late" else 201501
        orig.append({
            "loan_id": loan_id, "first_payment_date": str(fp), "credit_score": str(int(rng.integers(620, 820))),
            "ltv": str(int(rng.integers(50, 95))), "cltv": str(int(rng.integers(50, 95))),
            "dti": str(int(rng.integers(15, 50))), "orig_upb": "200000", "orig_rate": f"{rng.uniform(3.5, 4.5):.3f}",
            "mi_pct": "0", "num_units": "1", "orig_term": "360", "num_borrowers": "2",
            "first_time_homebuyer": "N", "occupancy": "P", "channel": "R", "loan_purpose": "P",
            "property_type": "SF", "state": str(rng.choice(["TX", "CA", "FL", "NY"])),
            "prepay_penalty_flag": "N", "super_conforming": "N", "harp_flag": "N",
        })
    perf = []
    for (loan_id, name), o in zip(loans, orig, strict=True):
        perf += _scenario_rows(name, loan_id, 2017 * 12 if o["first_payment_date"] == "201701" else FP_IDX)
    return loans, pd.DataFrame(orig), pd.DataFrame(perf)


def _to_positional(df, schema_cols):
    width = max(c["pos"] for c in schema_cols) + 1
    out = pd.DataFrame("", index=df.index, columns=range(width))
    for c in schema_cols:
        if c["name"] in df.columns:
            out[c["pos"]] = df[c["name"]].fillna("")
    return out


def write_raw(directory, schema, orig, perf, vintage=2015):
    directory.mkdir(parents=True, exist_ok=True)
    o_path, p_path = directory / f"sample_orig_{vintage}.txt", directory / f"sample_perf_{vintage}.txt"
    _to_positional(orig, schema["origination"]).to_csv(o_path, sep="|", header=False, index=False)
    _to_positional(perf, schema["performance"]).to_csv(p_path, sep="|", header=False, index=False)
    return o_path, p_path


def make_model_table(cfg, n=4000, seed=0):
    """Random modeling table with the configured feature columns and a target that depends on a few of them."""
    from creditrisk.config import feature_lists

    rng = np.random.default_rng(seed)
    spec = feature_lists(cfg, "full")
    df = pd.DataFrame({"loan_id": [f"L{i:06d}" for i in range(n)], "pred_period": "201612"})
    df["credit_score"] = rng.normal(740, 50, n).clip(600, 830).round()
    df["ltv"] = rng.normal(75, 15, n).clip(20, 97).round()
    df["cltv"] = (df.ltv + rng.integers(0, 5, n)).clip(upper=105)
    df["dti"] = np.where(rng.random(n) < 0.08, np.nan, rng.normal(34, 9, n).clip(5, 65).round())
    df["orig_upb"] = rng.uniform(50000, 450000, n).round(-3)
    df["orig_rate"] = rng.normal(4.0, 0.3, n).round(3)
    df["mi_pct"], df["num_units"], df["orig_term"] = 0.0, 1.0, 360.0
    df["num_borrowers"] = rng.choice([1.0, 2.0], n)
    df["upb_ratio"] = rng.uniform(0.8, 0.99, n)
    df["delinq_at_pred"] = rng.choice([0.0, 1.0, 2.0], n, p=[0.96, 0.03, 0.01])
    df["max_delinq_prior12"] = np.maximum(df.delinq_at_pred, rng.choice([0.0, 1.0, 3.0], n, p=[0.93, 0.05, 0.02]))
    df["ever_90_before"] = rng.choice([0, 1], n, p=[0.99, 0.01])
    for c in spec["categorical"]:
        df[c] = rng.choice(["A", "B", "C"], n, p=[0.6, 0.3, 0.1])
    df["state"] = rng.choice(["TX", "CA", "FL", "NY"], n)
    df["ead"] = df.orig_upb * df.upb_ratio
    z = (-4.6 + 0.04 * (df.ltv - 75) - 0.01 * (df.credit_score - 740)
         + 1.0 * df.delinq_at_pred + 0.3 * df.max_delinq_prior12)
    df["target"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    df["meta_disaster_event"] = ((df.target == 1) & (rng.random(n) < 0.3)).astype(int)
    df["meta_assist_event"] = ((df.target == 1) & (rng.random(n) < 0.5)).astype(int)
    return df
