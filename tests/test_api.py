import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from creditrisk.api import create_app
from creditrisk.logs import JsonFormatter
from creditrisk.serving import ModelLoadError, ScoringService, decide, top_reasons


def payload(row, features):
    out = {}
    for f in features:
        v = row[f]
        out[f] = None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
    return out


@pytest.fixture(scope="module", params=["scorecard", "lgbm"])
def kind(request):
    return request.param


@pytest.fixture(scope="module")
def client(kind, model_dirs):
    with TestClient(create_app(model_dirs[kind])) as c:
        yield c


@pytest.fixture(scope="module")
def service(kind, model_dirs):
    return ScoringService(model_dirs[kind])


@pytest.fixture
def loan(serving_table, service):
    return payload(serving_table.iloc[2500], service.features)


def test_health_ready_and_model_endpoints(client, kind):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}
    info = client.get("/v1/model").json()
    assert info["kind"] == kind and info["feature_set"] == "full" and info["name"] == f"creditrisk-full-{kind}"
    assert len(info["features"]) == 24 and info["decision_labels"] == ["no_action", "review", "escalate"]


def test_score_returns_pd_score_decision_and_reasons(client, loan):
    r = client.post("/v1/score", json={**loan, "loan_id": "L42"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 < body["pd"] < 1 and isinstance(body["score"], int)
    assert body["decision"] in ("no_action", "review", "escalate") and body["loan_id"] == "L42"
    reasons = body["reason_codes"]
    assert 1 <= len(reasons) <= 4
    assert [x["rank"] for x in reasons] == list(range(1, len(reasons) + 1))
    assert [x["impact"] for x in reasons] == sorted((x["impact"] for x in reasons), reverse=True)
    assert all(x["code"].startswith("R") and x["impact"] > 0 for x in reasons)
    assert body["model"]["kind"] in ("scorecard", "lgbm") and body["request_id"]


def test_api_result_equals_the_saved_model_and_the_shared_scoring_path(client, service, loan, serving_table):
    body = client.post("/v1/score", json=loan).json()
    row = serving_table.iloc[[2500]]
    assert body["pd"] == pytest.approx(service.scorer.predict_proba(row)[0, 1], rel=1e-12)
    scored = service.score(row).iloc[0]
    assert body["score"] == scored["score"] and body["decision"] == scored["decision"]
    assert [r["feature"] for r in body["reason_codes"]] == [f for f, _ in scored["reasons"]]


def test_worse_credit_score_raises_pd_and_its_reason_impact(client, loan):
    good = client.post("/v1/score", json={**loan, "credit_score": 800}).json()
    bad = client.post("/v1/score", json={**loan, "credit_score": 560}).json()
    assert bad["pd"] > good["pd"] and bad["score"] < good["score"]


@pytest.mark.parametrize("change", [
    {"credit_score": 1000}, {"credit_score": 120}, {"ltv": 0}, {"dti": 140}, {"occupancy": "X"},
    {"state": "Texas"}, {"orig_rate": float("nan")}, {"credit_score": "high"}, {"typo_field": 1},
])
def test_invalid_input_is_rejected_with_422(client, loan, change):
    r = client.post("/v1/score", content=json.dumps({**loan, **change}), headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert r.json()["request_id"] and all("input" not in e for e in r.json()["detail"])


def test_missing_required_field_is_rejected_but_nullable_fields_may_be_omitted(client, loan):
    required = {k: v for k, v in loan.items() if k != "orig_upb"}
    assert client.post("/v1/score", json=required).status_code == 422
    optional = {k: v for k, v in loan.items() if k != "dti"}
    assert client.post("/v1/score", json=optional).status_code == 200
    nulls = {**loan, "dti": None, "credit_score": None}
    assert client.post("/v1/score", json=nulls).status_code == 200


def test_unseen_state_in_valid_format_still_scores(client, loan):
    assert client.post("/v1/score", json={**loan, "state": "ZZ"}).status_code == 200


def test_request_id_is_echoed_or_generated(client, loan):
    r = client.post("/v1/score", json=loan, headers={"X-Request-ID": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123" and r.json()["request_id"] == "abc-123"
    r = client.post("/v1/score", json=loan, headers={"X-Request-ID": "bad id with spaces!"})
    assert r.headers["x-request-id"] != "bad id with spaces!" and len(r.headers["x-request-id"]) == 32


def test_request_logs_are_structured_and_hold_no_applicant_values(client, loan, caplog):
    caplog.set_level(logging.INFO, logger="creditrisk.api")
    marker = {**loan, "orig_upb": 123457.0, "loan_id": "SECRET-LOAN-ID"}
    client.post("/v1/score", json=marker)
    client.post("/v1/score", json={**marker, "credit_score": 9999})
    lines = [JsonFormatter().format(r) for r in caplog.records if getattr(r, "fields", {}).get("event") == "request"]
    assert len(lines) == 2
    ok, bad = (json.loads(x) for x in lines)
    assert ok["status"] == 200 and ok["latency_ms"] >= 0 and ok["decision"] and ok["request_id"]
    assert ok["path"] == "/v1/score"
    assert bad["status"] == 422 and bad["validation_errors"] == ["credit_score"]
    assert all(s not in "".join(lines) for s in ("123457", "SECRET-LOAN-ID", "9999"))


def test_health_checks_are_not_logged(client, caplog):
    caplog.set_level(logging.INFO, logger="creditrisk.api")
    client.get("/health")
    client.get("/ready")
    assert not [r for r in caplog.records if getattr(r, "fields", {}).get("event") == "request"]


def test_service_refuses_a_model_that_does_not_match_its_checksum(model_dirs, tmp_path):
    import shutil

    bad = tmp_path / "m"
    shutil.copytree(model_dirs["scorecard"], bad)
    with open(bad / "scorer.joblib", "ab") as fh:
        fh.write(b"tamper")
    with pytest.raises(ModelLoadError, match="checksum"):
        ScoringService(bad)
    with pytest.raises(ModelLoadError, match="incomplete"):
        ScoringService(tmp_path / "missing")


def test_service_requires_a_reason_code_for_every_feature(model_dirs, tmp_path):
    import shutil

    import yaml

    d = tmp_path / "m"
    shutil.copytree(model_dirs["scorecard"], d)
    cfg = yaml.safe_load((d / "project.yaml").read_text())
    del cfg["serving"]["reason_codes"]["dti"]
    (d / "project.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ModelLoadError, match="dti"):
        ScoringService(d)


def test_decision_bands_and_label_sets():
    p = np.array([0.001, 0.004, 0.01, 0.02, 0.5])
    assert list(decide(p, 0.004, 0.02, ["a", "b", "c"])) == ["a", "b", "b", "c", "c"]


def test_top_reasons_keeps_positive_impacts_largest_first():
    impact = np.array([[0.0, 5.0, 2.0, 9.0], [0.0, 0.0, 0.0, 0.0]])
    out = top_reasons(impact, ["a", "b", "c", "d"], 2)
    assert out[0] == [("d", 9.0), ("b", 5.0)] and out[1] == []


def test_scorecard_reason_impacts_sum_to_the_points_gap_from_the_best_case(model_dirs, serving_table):
    svc = ScoringService(model_dirs["scorecard"])
    df = serving_table.iloc[2000:2200].reset_index(drop=True)
    design = svc.impact.pipe[:-1].transform(df)
    impact = svc.impact(design)
    design = design.toarray() if hasattr(design, "toarray") else np.asarray(design)
    best_total = sum(svc.impact.best.values())
    got_total = (design * svc.impact.points).sum(axis=1)
    assert np.allclose(impact.sum(axis=1), best_total - got_total)


def test_example_requests_in_the_repo_are_valid_for_the_models_they_target(model_dirs, kind):
    root = Path(__file__).resolve().parents[1] / "examples"
    full = json.loads((root / "score_request_full.json").read_text())
    orig = json.loads((root / "score_request_origination.json").read_text())
    with TestClient(create_app(model_dirs[kind])) as c:
        r = c.post("/v1/score", json=full)
        assert r.status_code == 200 and r.json()["loan_id"] == "EXAMPLE-0001"
        assert c.post("/v1/score", json=orig).status_code == 422
    svc = ScoringService(model_dirs[kind])
    cfg = svc.cfg["features"]
    origination = set(cfg["origination"]["numeric"] + cfg["origination"]["categorical"])
    assert set(orig) - {"loan_id"} == origination and set(full) - {"loan_id"} == set(svc.features)


def test_container_healthcheck_exits_nonzero_when_nothing_is_listening(monkeypatch):
    from creditrisk import healthcheck

    monkeypatch.setenv("PORT", "1")
    with pytest.raises(SystemExit) as e:
        healthcheck.main()
    assert e.value.code == 1
