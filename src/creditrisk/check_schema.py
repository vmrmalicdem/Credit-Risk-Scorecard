"""Verify the positional schema map against the real files, and print Day-1 counts.

Usage (from repo root):
    py src/creditrisk/check_schema.py --orig data/raw/sample_orig_2015.txt --perf data/raw/sample_perf_2015.txt
Needs: pandas, pyyaml. Reads only the performance columns it needs to keep memory low.
"""
import argparse
import re

import pandas as pd
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--orig", required=True)
ap.add_argument("--perf", required=True)
ap.add_argument("--schema", default="configs/schema_map.yaml")
args = ap.parse_args()

schema = yaml.safe_load(open(args.schema))
omap = {c["pos"]: c["name"] for c in schema["origination"]}
pmap = {c["pos"]: c["name"] for c in schema["performance"]}

results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {label} {detail}")


def read(path, mapping, names=None):
    pos = sorted(p for p in mapping if names is None or mapping[p] in names)
    df = pd.read_csv(path, sep="|", header=None, dtype=str, usecols=pos,
                     keep_default_na=False, na_values=[""], low_memory=False)
    return df.rename(columns=mapping)


# ---------- origination ----------
o = read(args.orig, omap)
print(f"\nOrigination: {len(o):,} rows, {o.shape[1]} cols mapped")
check("loan_id unique", o.loan_id.is_unique)
check("loan_id pattern", o.loan_id.str.fullmatch(r"[FA]\d{2}Q[1-4]\d{7}").all(),
      f"(example: {o.loan_id.iloc[0]})")
fico = pd.to_numeric(o.credit_score, errors="coerce")
check("credit_score in 300-850 or 9999/blank", (fico.dropna().between(300, 850) | (fico.dropna() == 9999)).all())
ltv = pd.to_numeric(o.ltv, errors="coerce")
check("ltv in 1-998 or 999/blank", (ltv.dropna().between(1, 999)).all(),
      f"(median {ltv[ltv < 999].median():.0f})")
check("first_payment_date is YYYYMM", o.first_payment_date.str.fullmatch(r"\d{6}").all())
check("occupancy in P/I/S/9", o.occupancy.isin(list("PIS9")).all())
check("state is 2 letters", o.state.str.fullmatch(r"[A-Z]{2}").all())
check("amort_type all FRM (standard dataset)", (o.amort_type == "FRM").all(),
      f"(values: {o.amort_type.value_counts().to_dict()})")
print("orig_term values:", o.orig_term.value_counts().head(5).to_dict())
print("first_payment_date range:", o.first_payment_date.min(), "to", o.first_payment_date.max())
for c in ["vantage_score", "valuation_method", "special_program", "harp_flag"]:
    print(f"  missing/blank share of {c}: {o[c].isna().mean():.0%}")

# ---------- performance ----------
need = ["loan_id", "period", "current_upb", "delinq_status", "loan_age", "zero_bal_code",
        "actual_loss", "zero_bal_removal_upb", "current_rate", "mod_flag"]
p = read(args.perf, pmap, need)
print(f"\nPerformance: {len(p):,} rows (selected columns)")
check("every perf loan_id is in origination", p.loan_id.isin(set(o.loan_id)).all())
check("every orig loan has perf rows", set(o.loan_id) <= set(p.loan_id))
check("period is YYYYMM", p.period.str.fullmatch(r"\d{6}").all(),
      f"({p.period.min()} to {p.period.max()}; last period = effective performance cutoff for this sample)")
check("(loan_id, period) unique", not p.duplicated(["loan_id", "period"]).any())
ds = p.delinq_status.dropna()
check("delinq_status values plausible", ds.str.fullmatch(r"\d{2}|RA|XX").all(),
      f"(top values: {ds.value_counts().head(6).to_dict()})")
print("zero_bal_code distribution (rows):", p.zero_bal_code.value_counts(dropna=True).to_dict())

# ---------- Day-1 counts that decide feasibility ----------
num = pd.to_numeric(ds.where(ds.str.fullmatch(r"\d{2}")), errors="coerce")
sev = p.loc[ds.index].assign(n=num)
ever90 = set(sev.loc[(sev.n >= 3) | (sev.delinq_status == "RA"), "loan_id"])
credit_zb = set(p.loc[p.zero_bal_code.isin(["02", "03", "09", "15"]), "loan_id"])
has_loss = set(p.loc[p.actual_loss.notna(), "loan_id"])
rows_per_loan = p.groupby("loan_id").size()

print("\n--- Day-1 feasibility counts (whole life of the sample, not yet 12-month window) ---")
print(f"loans total:                         {o.shape[0]:,}")
print(f"loans ever 90+ days delinq or REO:   {len(ever90):,}")
print(f"loans with zero_bal_code 02/03/09/15: {len(credit_zb):,}")
print(f"loans with non-null actual_loss:     {len(has_loss):,}")
print(f"loans ever zero_bal_code 01 (payoff): {p.loc[p.zero_bal_code == '01', 'loan_id'].nunique():,}")
print(f"perf rows per loan: min {rows_per_loan.min()}, median {rows_per_loan.median():.0f}, max {rows_per_loan.max()}")

print(f"\n{sum(results)}/{len(results)} checks passed")
