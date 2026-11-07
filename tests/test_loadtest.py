import asyncio

import httpx
import pytest

from creditrisk.api import create_app
from creditrisk.loadtest import percentiles, run_load, sample_payloads
from creditrisk.serving import ScoringService


def test_percentiles_of_a_known_series():
    p = percentiles(list(range(1, 101)))
    assert p["p50"] == pytest.approx(50.5) and p["p99"] == pytest.approx(99.01) and p["max"] == 100


def test_load_run_against_the_app_counts_every_request_and_reports_latency(model_dirs, serving_table, tmp_path):
    svc = ScoringService(model_dirs["scorecard"])
    sample = tmp_path / "t.parquet"
    serving_table.to_parquet(sample, index=False)
    payloads = sample_payloads(sample, svc.features, 40)
    assert len(payloads) == 40 and set(payloads[0]) == set(svc.features)

    async def go():
        transport = httpx.ASGITransport(app=create_app(model_dirs["scorecard"]))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await run_load(client, payloads, 60, [1, 4], warmup=5)

    levels = asyncio.run(go())
    assert [lv["concurrency"] for lv in levels] == [1, 4]
    for lv in levels:
        assert lv["ok"] == 60 and lv["transport_errors"] == 0 and lv["requests_per_second"] > 0
        assert 0 < lv["latency_ms"]["p50"] <= lv["latency_ms"]["p99"] <= lv["latency_ms"]["max"]


def test_invalid_payloads_show_up_as_non_200_statuses(model_dirs, serving_table, tmp_path):
    svc = ScoringService(model_dirs["scorecard"])
    sample = tmp_path / "t.parquet"
    serving_table.to_parquet(sample, index=False)
    payloads = [{**p, "credit_score": 5} for p in sample_payloads(sample, svc.features, 5)]

    async def go():
        transport = httpx.ASGITransport(app=create_app(model_dirs["scorecard"]))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await run_load(client, payloads, 10, [2], warmup=0)

    (level,) = asyncio.run(go())
    assert level["ok"] == 0 and level["status_counts"] == {422: 10}
