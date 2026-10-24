"""Day 2 prep: count eligible loans and 12-month defaults under candidate prediction rules.

Usage (from repo root):
    py src/creditrisk/window_counts.py --orig data/raw/sample_orig_2015.txt --perf data/raw/sample_perf_2015.txt
    optional: --ages 12 24 36 --calendar 201712 201812 201912 --max-first-pay 201603

Default event (PROPOSED, see docs/target_definition.md): in a month after the prediction month, any of
  - delinquency status >= 03 (90+ days) or RA (REO acquisition)
  - zero balance code 02, 03, 09 or 15 (credit-event terminations)
Zero balance 01 (payoff) = non-default, censored.  16 and 96 = excluded if they end the loan inside the window.
Eligible at prediction month: row exists, status 00/01/02, not terminated, balance > 0.
Age is measured as (period - first payment month) + 1, matching the guide's non-modified loan-age formula.
"""
import argparse

import pandas as pd
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--orig", required=True)
ap.add_argument("--perf", required=True)
ap.add_argument("--schema", default="configs/schema_map.yaml")
ap.add_argument("--ages", type=int, nargs="*", default=[12, 24, 36])
ap.add_argument("--calendar", type=int, nargs="*", default=[201712, 201812, 201912])
ap.add_argument("--max-first-pay", type=int, default=None, help="drop loans with later first payment (YYYYMM)")
ap.add_argument("--window", type=int, default=12)
args = ap.parse_args()

schema = yaml.safe_load(open(args.schema))
omap = {c["pos"]: c["name"] for c in schema["origination"]}
pmap = {c["pos"]: c["name"] for c in schema["performance"]}


def read(path, mapping, names=None):
    pos = sorted(p for p in mapping if names is None or mapping[p] in names)
    df = pd.read_csv(path, sep="|", header=None, dtype=str, usecols=pos,
                     keep_default_na=False, na_values=[""], low_memory=False)
    return df.rename(columns=mapping)


def idx(yyyymm):  # YYYYMM -> month index
    v = pd.to_numeric(yyyymm)
    return (v // 100) * 12 + (v % 100) - 1


o = read(args.orig, omap, ["loan_id", "first_payment_date"])
fp = pd.to_numeric(o.first_payment_date)
print("first payment month, loans per year:", (fp // 100).value_counts().sort_index().to_dict())
print("loan_id vintage prefixes:", o.loan_id.str[:3].value_counts().head(6).to_dict())
if args.max_first_pay:
    o = o[fp <= args.max_first_pay]
    print(f"kept {len(o):,} loans with first payment <= {args.max_first_pay}")
o["fp_idx"] = idx(o.first_payment_date)

# sentinel shares (blank OR documented "Not Available" code) for origination fields
full_o = read(args.orig, omap)
print("\nFields that are mostly unavailable (blank or sentinel):")
for c in schema["origination"]:
    s = full_o[c["name"]]
    share = (s.isna() | s.isin(c.get("missing", []))).mean()
    if share > 0.2:
        print(f"  {c['name']}: {share:.0%}")

p = read(args.perf, pmap, ["loan_id", "period", "current_upb", "delinq_status", "zero_bal_code"])
p = p[p.loan_id.isin(set(o.loan_id))].copy()
p["pidx"] = idx(p.period)
st = p.delinq_status
num = pd.to_numeric(st.where(st.str.fullmatch(r"\d{2}", na=False)), errors="coerce")
p["sev90"] = (num >= 3) | (st == "RA")
p["credit_zb"] = p.zero_bal_code.isin(["02", "03", "09", "15"])
p["event"] = p.sev90 | p.credit_zb
p["status_ok"] = st.isin(["00", "01", "02"])
p["active_pos"] = p.zero_bal_code.isna() & (pd.to_numeric(p.current_upb, errors="coerce") > 0)

ev = p.loc[p.event, ["loan_id", "pidx"]].rename(columns={"pidx": "ev_idx"})
term = (p.loc[p.zero_bal_code.notna(), ["loan_id", "pidx", "zero_bal_code"]]
        .sort_values("pidx").drop_duplicates("loan_id", keep="last")
        .rename(columns={"pidx": "term_idx", "zero_bal_code": "term_code"}))
rows = p[["loan_id", "pidx", "status_ok", "active_pos"]]
W = args.window
COVID = (idx(pd.Series(["202003"])).iloc[0], idx(pd.Series(["202112"])).iloc[0])


def summarize(label, pred_idx):
    d = o[["loan_id"]].assign(pred=pred_idx.values)
    d = d.merge(rows, left_on=["loan_id", "pred"], right_on=["loan_id", "pidx"], how="inner")
    d = d[d.status_ok & d.active_pos][["loan_id", "pred"]]
    if d.empty:
        return dict(rule=label, eligible=0)
    e = d.merge(ev, on="loan_id", how="left")
    e["in_win"] = (e.ev_idx > e.pred) & (e.ev_idx <= e.pred + W)
    e["before"] = e.ev_idx <= e.pred
    g = e.groupby("loan_id").agg(pred=("pred", "first"), default=("in_win", "any"), prior90=("before", "any"))
    g = g.join(term.set_index("loan_id"), how="left")
    in_term_win = (g.term_idx > g.pred) & (g.term_idx <= g.pred + W)
    prepaid = in_term_win & (g.term_code == "01") & ~g.default
    excluded = in_term_win & g.term_code.isin(["16", "96"]) & ~g.default
    keep = ~excluded
    n = int(keep.sum())
    covid = ((g.pred + W >= COVID[0]) & (g.pred + 1 <= COVID[1]))[keep].mean()
    return dict(rule=label, eligible=n, defaults=int(g.default[keep].sum()),
                rate_pct=round(100 * g.default[keep].mean(), 2),
                prior_90_cured=int(g.prior90[keep].sum()), prepaid_in_window=int(prepaid.sum()),
                excluded_16_96=int(excluded.sum()),
                window_covid_overlap_pct=round(100 * covid), pred_min=int(g.pred.min()), pred_max=int(g.pred.max()))


out = []
for a in args.ages:
    out.append(summarize(f"age {a}", o.fp_idx + a - 1))
for c in args.calendar:
    out.append(summarize(f"calendar {c}", pd.Series(idx(pd.Series([str(c)])).iloc[0], index=o.index)))
res = pd.DataFrame(out)
for col in ("pred_min", "pred_max"):  # month index -> YYYYMM for readability
    if col in res:
        res[col] = res[col].map(lambda v: f"{int(v) // 12}{int(v) % 12 + 1:02d}" if pd.notna(v) else v)
print(f"\nWindow = {W} months after the prediction month\n")
print(res.to_string(index=False))
