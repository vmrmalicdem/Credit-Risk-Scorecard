"""Expected loss, LGD and stress-shock arithmetic."""
import numpy as np
import pandas as pd
import pytest

from creditrisk.ecl import (
    assign_lgd,
    expected_loss,
    lgd_table,
    prepare_liquidations,
    shock,
    wilson_interval,
)


def test_expected_loss_is_pd_times_lgd_times_ead():
    assert expected_loss([0.01, 0.02], [0.25, 0.5], [100_000, 200_000]) == pytest.approx(250 + 2000)
    assert expected_loss([0.0, 0.0], [0.3, 0.3], [1, 1]) == 0.0


def test_shock_raises_ltv_and_cltv_with_cap_and_leaves_input_untouched():
    d = pd.DataFrame({"ltv": [70.0, 140.0], "cltv": [75.0, 145.0], "orig_rate": [4.0, 5.0]})
    out = shock(d, ltv_pp=20, rate_pts=0.0, ltv_cap=150.0)
    assert out.ltv.tolist() == [90.0, 150.0] and out.cltv.tolist() == [95.0, 150.0]
    assert out.orig_rate.tolist() == [4.0, 5.0]
    assert d.ltv.tolist() == [70.0, 140.0]


def test_shock_rate_only_changes_rate():
    d = pd.DataFrame({"ltv": [70.0], "cltv": [75.0], "orig_rate": [4.0]})
    out = shock(d, 0, 2.0, 150.0)
    assert out.orig_rate.iloc[0] == 6.0 and out.ltv.iloc[0] == 70.0


def test_zero_shock_is_identity():
    d = pd.DataFrame({"ltv": [70.0], "cltv": [75.0], "orig_rate": [4.0]})
    pd.testing.assert_frame_equal(shock(d, 0, 0.0, 150.0), d)


def _liq(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "loan_id": [f"L{i}" for i in range(n)], "actual_loss": rng.uniform(-5000, 190000, n),
        "zero_bal_removal_upb": rng.uniform(80000, 200000, n), "ltv": rng.uniform(50, 95, n)
    }).assign(lgd_raw=lambda d: d.actual_loss / d.zero_bal_removal_upb)


def test_lgd_is_floored_and_capped_and_rows_without_loss_are_dropped():
    liq = _liq()
    liq.loc[0, "actual_loss"] = np.nan
    liq.loc[1, "lgd_raw"] = 1.7
    liq.loc[2, "lgd_raw"] = -0.4
    out = prepare_liquidations(liq, ltv_split=80.0, floor=0.0, cap=1.0)
    assert len(out) == len(liq) - 1
    assert out.lgd.between(0, 1).all()
    assert out.set_index("loan_id").lgd["L1"] == 1.0 and out.set_index("loan_id").lgd["L2"] == 0.0


def test_pooled_lgd_is_the_mean_of_clipped_values():
    liq = prepare_liquidations(_liq(), 80.0, 0.0, 1.0)
    table, use_seg, _, pooled = lgd_table(liq, "pooled", 5, np.random.default_rng(0), 50)
    assert not use_seg
    assert pooled == pytest.approx(liq.lgd.mean())
    row = table[table.segment == "all"].iloc[0]
    assert row.ci_low <= row.mean_lgd <= row.ci_high and row.used_in_ecl


def test_segmented_lgd_needs_enough_loans_in_each_segment():
    liq = prepare_liquidations(_liq(), 80.0, 0.0, 1.0)
    _, used, _, _ = lgd_table(liq, "segmented", 5, np.random.default_rng(0), 50)
    assert used
    _, used_small, _, _ = lgd_table(liq, "segmented", 10_000, np.random.default_rng(0), 50)
    assert not used_small


def test_assign_lgd_pooled_and_segmented():
    ltv = np.array([60.0, 90.0])
    assert assign_lgd(ltv, False, {}, 0.23, 80.0).tolist() == [0.23, 0.23]
    out = assign_lgd(ltv, True, {"ltv_gt_80": 0.2, "ltv_le_80": 0.3}, 0.23, 80.0)
    assert out.tolist() == [0.3, 0.2]


def test_wilson_interval_known_values_and_bounds():
    lo, hi = wilson_interval(29, 221)
    assert lo == pytest.approx(0.0929, abs=5e-4) and hi == pytest.approx(0.1819, abs=5e-4)
    lo0, hi0 = wilson_interval(0, 50)
    assert lo0 == pytest.approx(0.0, abs=1e-12) and 0 < hi0 < 0.1
