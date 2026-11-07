import json

import numpy as np
import pandas as pd
import pytest

from creditrisk import batch
from creditrisk.serving import ScoringService
from creditrisk.validation import DataValidationError


@pytest.fixture(scope="module", params=["scorecard", "lgbm"])
def kind(request):
    return request.param


@pytest.fixture
def loans(serving_table):
    df = serving_table.iloc[2000:2400].reset_index(drop=True).copy()
    df["loan_id"] = [f"B{i:05d}" for i in range(len(df))]
    return df


def test_batch_output_equals_the_service_for_every_loan(kind, model_dirs, loans, tmp_path):
    src = tmp_path / "in.parquet"
    loans.to_parquet(src, index=False)
    summary = batch.run(model_dirs[kind], src, tmp_path / "out.parquet", chunk_size=150, log=lambda *_: None)
    out = pd.read_parquet(tmp_path / "out.parquet")
    svc = ScoringService(model_dirs[kind])
    ref = svc.score(loans)
    assert summary["rows_scored"] == len(loans) and summary["rows_rejected"] == 0 and summary["status"] == "ok"
    assert list(out.loan_id) == list(loans.loan_id) and list(out.row) == list(range(len(loans)))
    assert np.allclose(out.pd, ref.pd, rtol=1e-12, atol=0) and (out.score.to_numpy() == ref.score.to_numpy()).all()
    assert list(out.decision) == list(ref.decision)
    first = [svc.codes[r[0][0]]["code"] if r else "" for r in ref.reasons]
    assert list(out.reason_code_1) == first
    assert set(out.model_name) == {f"creditrisk-full-{kind}"} and set(out.model_version) == {1}


def test_chunk_size_does_not_change_results(model_dirs, loans, tmp_path):
    src = tmp_path / "in.csv"
    loans.to_csv(src, index=False)
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    batch.run(model_dirs["lgbm"], src, a, chunk_size=7, log=lambda *_: None)
    batch.run(model_dirs["lgbm"], src, b, chunk_size=100_000, log=lambda *_: None)
    cols = ["loan_id", "row", "pd", "score", "decision", "reason_code_1", "reason_code_2"]
    pd.testing.assert_frame_equal(pd.read_csv(a)[cols], pd.read_csv(b)[cols])


def test_invalid_rows_are_quarantined_not_scored_and_the_rest_still_score(model_dirs, loans, tmp_path):
    bad = loans.copy()
    bad.loc[3, "credit_score"] = 1200
    bad.loc[10, "occupancy"] = "Q"
    bad.loc[11, "dti"] = -5.0
    src = tmp_path / "in.parquet"
    bad.to_parquet(src, index=False)
    s = batch.run(model_dirs["scorecard"], src, tmp_path / "out.parquet", chunk_size=64, log=lambda *_: None)
    out = pd.read_parquet(tmp_path / "out.parquet")
    rej = pd.read_csv(tmp_path / "out_rejects.csv")
    assert s["rows_rejected"] == 3 and s["rows_scored"] == len(loans) - 3
    assert sorted(rej.row.unique()) == [3, 10, 11] and set(rej.loan_id) == {"B00003", "B00010", "B00011"}
    assert not {3, 10, 11} & set(out.row)
    assert {"credit_score", "occupancy", "dti"} <= set(rej.column)
    assert {c["column"] for c in s["reject_causes"]} == {"credit_score", "occupancy", "dti"}


def test_too_many_rejects_flags_the_run(model_dirs, loans, tmp_path):
    bad = loans.assign(credit_score=1200)
    src = tmp_path / "in.parquet"
    bad.to_parquet(src, index=False)
    s = batch.run(model_dirs["scorecard"], src, tmp_path / "out.parquet", log=lambda *_: None)
    assert s["status"] == "rejects_above_threshold" and s["rows_scored"] == 0
    with pytest.raises(SystemExit) as e:
        batch.main(["--input", str(src), "--output", str(tmp_path / "o2.parquet"),
                    "--model-dir", str(model_dirs["scorecard"])])
    assert e.value.code == 2


def test_missing_feature_column_aborts_and_leaves_no_output(model_dirs, loans, tmp_path):
    src = tmp_path / "in.parquet"
    loans.drop(columns=["dti"]).to_parquet(src, index=False)
    with pytest.raises(DataValidationError, match="dti"):
        batch.run(model_dirs["scorecard"], src, tmp_path / "out.parquet", log=lambda *_: None)
    assert not (tmp_path / "out.parquet").exists() and not (tmp_path / "out.parquet.partial").exists()


def test_summary_reports_decision_bands_and_observed_default_rates(model_dirs, loans, tmp_path):
    src = tmp_path / "in.parquet"
    loans.to_parquet(src, index=False)
    batch.run(model_dirs["scorecard"], src, tmp_path / "out.parquet", log=lambda *_: None)
    s = json.loads((tmp_path / "out_summary.json").read_text())
    assert s["input"]["rows"] == len(loans) and len(s["input"]["sha256"]) == 64
    assert sum(v["loans"] for v in s["decisions"].values()) == len(loans)
    obs = s["observed_default_rate_by_decision"]
    assert sum(v["defaults"] for v in obs.values()) == int(loans.target.sum())
    assert s["model"]["scorer_sha256"] and s["rows_per_second"] > 0 and 0 < s["pd"]["mean"] < 1
