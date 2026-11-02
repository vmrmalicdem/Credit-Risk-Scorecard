"""Target definition, eligibility and funnel logic, checked against hand-built loan scenarios."""
import numpy as np
import pytest
from synth import SCENARIOS


def scenario_loans(synthetic, name):
    return [lid for lid, n in synthetic["loans"] if n == name]


@pytest.mark.parametrize("name", [n for n, v in SCENARIOS.items() if v[0]])
def test_eligible_scenarios_are_in_the_model_table_with_expected_target(synthetic, name):
    model = synthetic["model"].set_index("loan_id")
    for lid in scenario_loans(synthetic, name):
        assert lid in model.index
        assert model.loc[lid, "target"] == SCENARIOS[name][1], name


@pytest.mark.parametrize("name", [n for n, v in SCENARIOS.items() if not v[0]])
def test_ineligible_scenarios_are_absent(synthetic, name):
    model = synthetic["model"].set_index("loan_id")
    assert not any(lid in model.index for lid in scenario_loans(synthetic, name))


def test_funnel_counts_match_the_construction(synthetic):
    f = dict(zip(synthetic["funnel"].step, synthetic["funnel"].loans, strict=True))
    n_total = len(synthetic["loans"])
    assert f["origination loans in sample file"] == n_total
    assert f.get("after dropping first payment later than 201603") == n_total - 1
    assert f["with a performance row at age 24"] == n_total - 2
    assert f["eligible: status < 90 days, active, balance > 0"] == n_total - 3
    assert f["excluded: ended in window as code 16 / 96"] == 1
    assert f["  (kept) paid off inside window, counted as non-default"] == 1
    assert f["MODELING POPULATION"] == n_total - 4
    assert f["  defaults in window (target = 1)"] == int(synthetic["model"].target.sum()) == 3


def test_one_row_per_loan_binary_target_positive_ead(synthetic):
    m = synthetic["model"]
    assert m.loan_id.is_unique
    assert set(m.target.unique()) <= {0, 1}
    assert (m.ead > 0).all()


def test_ead_is_balance_at_the_prediction_month_not_later(synthetic):
    m = synthetic["model"].set_index("loan_id")
    assert m.loc["S00", "ead"] == pytest.approx(200000 - 400 * 23)


def test_behaviour_features_describe_the_state_at_prediction(synthetic):
    m = synthetic["model"].set_index("loan_id")
    assert m.loc["S04", "delinq_at_pred"] == 1 and m.loc["S04", "max_delinq_prior12"] == 1
    assert m.loc["S10", "ever_90_before"] == 1 and m.loc["S10", "delinq_at_pred"] == 0
    assert m.loc["S00", "ever_90_before"] == 0
    assert m.loc["S11", "mod_at_pred"] == "Y" and m.loc["S00", "mod_at_pred"] == "N"
    assert m.loc["S00", "upb_ratio"] == pytest.approx((200000 - 400 * 23) / 200000)


def test_disaster_and_assistance_flags_are_reporting_only_columns(synthetic):
    m = synthetic["model"].set_index("loan_id")
    assert m.loc["S01", "meta_disaster_event"] == 1
    assert m.loc["S00", "meta_disaster_event"] == 0
    assert list(m.columns).index("meta_disaster_event") > list(m.columns).index("mod_at_pred")


def test_liquidation_table_holds_credit_event_with_realized_lgd(synthetic):
    z = synthetic["liq"]
    assert list(z.loan_id) == ["S07"]
    assert z.lgd_raw.iloc[0] == pytest.approx(30000 / 180000)
    assert bool(z.in_model_population.iloc[0])


def test_sentinel_codes_become_missing(synthetic, cfg, schema, tmp_path):
    from synth import write_raw

    from creditrisk.build_dataset import build

    orig = synthetic["orig"].copy()
    orig.loc[orig.loan_id == "S00", "credit_score"] = "9999"
    orig.loc[orig.loan_id == "S00", "dti"] = "999"
    o_path, p_path = write_raw(tmp_path, schema, orig, synthetic["perf"])
    model, _, _ = build(cfg, schema, o_path, p_path)
    row = model.set_index("loan_id").loc["S00"]
    assert np.isnan(row.credit_score) and np.isnan(row.dti)
