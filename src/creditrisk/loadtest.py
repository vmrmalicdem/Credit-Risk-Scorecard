"""Closed-loop load test for the scoring API: N requests at a fixed concurrency, latency percentiles per level.

Usage:  py -m creditrisk.loadtest --url http://127.0.0.1:8000 --requests 1000 --concurrency 1 4 16
Request bodies are real loans sampled from the modeling table (--sample), restricted to the loaded model's features.
Each worker sends its next request as soon as the previous one returns, so latency here is service time under that
level of concurrency, measured from the client on the same machine (no network). Treat the numbers as a baseline for
this hardware and this configuration, not as a capacity guarantee. Writes reports/loadtest.json.
"""
import argparse
import asyncio
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd


def sample_payloads(path, features, n, seed=42):
    df = pd.read_csv(path) if str(path).lower().endswith(".csv") else pd.read_parquet(path)
    df = df.sample(min(n, len(df)), random_state=seed)[features]
    return [{k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()}
            for row in df.to_dict("records")]


def percentiles(ms):
    a = np.asarray(ms)
    return {"mean": float(a.mean()), "p50": float(np.quantile(a, 0.50)), "p90": float(np.quantile(a, 0.90)),
            "p95": float(np.quantile(a, 0.95)), "p99": float(np.quantile(a, 0.99)), "max": float(a.max())}


async def run_level(client, payloads, requests, concurrency):
    queue = asyncio.Queue()
    for i in range(requests):
        queue.put_nowait(payloads[i % len(payloads)])
    latencies, statuses, errors = [], {}, 0

    async def worker():
        nonlocal errors
        while True:
            try:
                body = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            t = time.perf_counter()
            try:
                r = await client.post("/v1/score", json=body)
                statuses[r.status_code] = statuses.get(r.status_code, 0) + 1
            except httpx.HTTPError:
                errors += 1
                continue
            latencies.append((time.perf_counter() - t) * 1000)

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.perf_counter() - started
    ok = statuses.get(200, 0)
    return {"concurrency": concurrency, "requests": requests, "ok": ok, "status_counts": statuses,
            "transport_errors": errors, "seconds": round(wall, 2),
            "requests_per_second": round(len(latencies) / wall, 1),
            "latency_ms": percentiles(latencies) if latencies else None}


async def run_load(client, payloads, requests, levels, warmup=50):
    for body in payloads[:warmup] if warmup else []:
        await client.post("/v1/score", json=body)
    return [await run_level(client, payloads, requests, c) for c in levels]


def print_table(results):
    head = ("conc", "req/s", "mean", "p50", "p90", "p95", "p99", "max", "ok", "errors")
    print(" ".join(f"{h:>8}" for h in head), "  (latency in ms)")
    for r in results:
        lat = r["latency_ms"]
        bad = r["requests"] - r["ok"]
        cells = [r["concurrency"], r["requests_per_second"], *(f"{lat[k]:.1f}" for k in
                 ("mean", "p50", "p90", "p95", "p99", "max")), r["ok"], bad]
        print(" ".join(f"{c:>8}" for c in cells))


async def amain(args):
    async with httpx.AsyncClient(base_url=args.url, timeout=30,
                                 limits=httpx.Limits(max_connections=max(args.concurrency))) as client:
        info = (await client.get("/v1/model")).json()
        payloads = sample_payloads(args.sample, info["features"], args.pool)
        results = await run_load(client, payloads, args.requests, args.concurrency, args.warmup)
    return info, results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--requests", type=int, default=1000, help="requests per concurrency level")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--pool", type=int, default=1000, help="number of distinct loans to sample")
    ap.add_argument("--sample", default="data/processed/model_table.parquet")
    ap.add_argument("--output", default="reports/loadtest.json")
    args = ap.parse_args(argv)
    info, results = asyncio.run(amain(args))
    print_table(results)
    report = {"run_utc": datetime.now(UTC).isoformat(timespec="seconds"), "url": args.url,
              "model": {k: info[k] for k in ("name", "version", "kind", "feature_set")},
              "client_machine": {"platform": platform.platform(), "python": platform.python_version(),
                                 "cpus": os.cpu_count()},
              "requests_per_level": args.requests, "levels": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
