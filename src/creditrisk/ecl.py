"""Simplified 12-month expected loss: realized LGD, conversion share, feature shocks and the ECL sum.

ECL = PD x LGD x EAD summed over loans. Not IFRS 9: no staging, lifetime PD or discounting.
"""
import numpy as np
import pandas as pd


def prepare_liquidations(liq, ltv_split, floor, cap):
    liq = liq[liq.actual_loss.notna() & (liq.zero_bal_removal_upb > 0)].copy()
    liq["lgd"] = liq.lgd_raw.clip(floor, cap)
    liq["segment"] = np.where(liq.ltv > ltv_split, "ltv_gt_80", "ltv_le_80")
    return liq


def segment_stats(g, name, rng, n_boot):
    x = g.lgd.to_numpy()
    boot = [rng.choice(x, len(x)).mean() for _ in range(n_boot)]
    return {"segment": name, "n": len(x), "mean_lgd": x.mean(), "median_lgd": float(np.median(x)),
            "ci_low": np.quantile(boot, 0.025), "ci_high": np.quantile(boot, 0.975),
            "mean_lgd_balance_weighted": np.average(x, weights=g.zero_bal_removal_upb),
            "mean_raw_unclipped": g.lgd_raw.mean(), "n_floored": int((g.lgd_raw < 0).sum()),
            "n_capped": int((g.lgd_raw > 1).sum())}


def lgd_table(liq, mode, min_segment, rng, n_boot):
    """Per-segment and pooled LGD statistics; returns (table, use_segments, segment -> mean LGD, pooled LGD)."""
    segs = [segment_stats(g, s, rng, n_boot) for s, g in liq.groupby("segment")]
    use_segments = mode == "segmented" and len(segs) == 2 and all(s["n"] >= min_segment for s in segs)
    table = pd.DataFrame(segs + [segment_stats(liq, "all", rng, n_boot)])
    table.insert(1, "used_in_ecl", [use_segments] * len(segs) + [not use_segments])
    lgd_map = {s["segment"]: s["mean_lgd"] for s in segs}
    pooled = float(table.loc[table.segment == "all", "mean_lgd"].iloc[0])
    return table, use_segments, lgd_map, pooled


def assign_lgd(ltv, use_segments, lgd_map, pooled, ltv_split):
    if use_segments:
        return np.where(ltv > ltv_split, lgd_map["ltv_gt_80"], lgd_map["ltv_le_80"])
    return np.full(len(ltv), pooled)


def wilson_interval(k, n, z=1.96):
    """Wilson score interval for a proportion k / n; returns (low, high)."""
    p = k / n
    den = 1 + z ** 2 / n
    ctr = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return ctr - half, ctr + half


def shock(d, ltv_pp, rate_pts, ltv_cap):
    """Copy of d with LTV and CLTV raised by ltv_pp points (capped) and note rate raised by rate_pts points."""
    d = d.copy()
    if ltv_pp:
        d["ltv"] = (d.ltv + ltv_pp).clip(upper=ltv_cap)
        d["cltv"] = (d.cltv + ltv_pp).clip(upper=ltv_cap)
    if rate_pts:
        d["orig_rate"] = d.orig_rate + rate_pts
    return d


def expected_loss(pd_prob, lgd, ead):
    return float((np.asarray(pd_prob) * np.asarray(lgd) * np.asarray(ead)).sum())
