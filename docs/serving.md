# Serving: API, container, batch scoring and load test

One exported model folder feeds everything: the API, the batch scorer and the Docker image. All three use the same
`ScoringService` (`src/creditrisk/serving.py`), so a loan gets the same PD, score, decision and reasons whichever way
it is scored.

## 1. Export a model

```
py -m creditrisk.export --name creditrisk-full-scorecard --alias production --out deploy/model
```

This writes `scorer.joblib`, `project.yaml` (the configuration the model was exported with, including the decision
cutoffs and reason codes) and `metadata.json` (registry name and version, run, git commit, data fingerprint and the
SHA-256 of the scorer). The server refuses to start if `scorer.joblib` no longer matches that checksum. `deploy/` is
git-ignored: the model was fitted on loan data. Use `--local artifacts/full/scorecard.joblib` to export without the
registry.

Which model to serve:

| Model | Scores | Decision labels |
|---|---|---|
| `creditrisk-full-*` | existing loans at month 24 (uses payment history) | `no_action`, `review`, `escalate` |
| `creditrisk-origination-*` | applications (origination fields only) | `approve`, `refer`, `decline` |

The scorecard gives the cleanest reason codes. The LightGBM model also works; its reasons are positive TreeSHAP
contributions.

## 2. The API

```
set MODEL_DIR=deploy/model        (PowerShell: $env:MODEL_DIR="deploy/model")
py -m creditrisk.api              (docs at http://127.0.0.1:8000/docs)
curl -X POST http://127.0.0.1:8000/v1/score -H "Content-Type: application/json" -d @examples/score_request_full.json
```

`POST /v1/score` returns:

```json
{"request_id": "...", "loan_id": "EXAMPLE-0001", "pd": 0.0051, "score": 600, "decision": "review",
 "reason_codes": [{"rank": 1, "code": "R02", "feature": "ltv", "description": "Loan-to-value ratio at origination",
                   "value": 80.0, "impact": 13.39}],
 "model": {"name": "creditrisk-full-scorecard", "version": 1, "kind": "scorecard", "feature_set": "full"}}
```

- `pd` is the calibrated probability of default in the 12 months after the prediction month.
- `score` is on the points scale in `configs/project.yaml` (600 at 200:1 odds, 20 points to double the odds); higher is
  lower risk.
- `decision` comes from two PD cutoffs in the config (`serving.decision`). The defaults are illustrative policy, not a
  model output. Set them from the decision bands that the batch summary reports on your data.
- `reason_codes` are the features that cost the most risk, largest first, with the code and text from the config. For
  the scorecard `impact` is points lost against the best bin of that feature; for LightGBM it is the TreeSHAP
  contribution in log-odds. Both are on the model's raw scale, before the small Platt recalibration.

**Input validation.** The request schema is built at start-up from the model's feature list and the same ranges and
allowed values the training data was checked against (`validation.columns`). Out-of-range numbers, unknown codes, a
bad state format, missing required fields and unknown field names all return 422 with the failing fields. Unknown
fields are rejected on purpose: a misspelled field would otherwise be scored as a missing value. Fields that were
allowed to be missing in training (for example `dti`) may be omitted or null.

Other endpoints: `GET /health` (process is up), `GET /ready` (model loaded and warmed up), `GET /v1/model` (name,
version, features, cutoffs and lineage).

**Logging.** One JSON object per line on stdout: timestamp, level, event, `request_id` (the `X-Request-ID` header if
given, else generated, and echoed in the response), path, status, latency in ms, decision, PD, score and model
version. Applicant field values and loan ids are never logged. Rejected requests log only the names of the failing
fields.

## 3. Docker

```
py -m creditrisk.export --name creditrisk-full-scorecard --alias production --out deploy/model
docker build -t credit-risk-api .
docker run --rm -p 8000:8000 credit-risk-api
docker ps          (STATUS shows "healthy" once /ready answers)
```

The image is `python:3.13-slim` with the pinned runtime dependencies in `requirements-serve.txt` (no MLflow, no dev
tools), the `creditrisk` package, and the exported model folder. It installs `libgomp1` (LightGBM needs it), runs as a
non-root user, and the `HEALTHCHECK` calls `python -m creditrisk.healthcheck`, which exits 0 only when `/ready`
returns 200. Scale out with `-e WORKERS=<n>` (one process per CPU core) or run several containers behind a load
balancer. Single-threaded math libraries are set (`OMP_NUM_THREADS=1`) so workers do not fight over cores.

To change the served model, export a new folder and rebuild. The image therefore carries one fixed model version
whose metadata is visible at `/v1/model`.

## 4. Batch scoring

```
py -m creditrisk.batch --input loans.parquet --output scores.parquet --model-dir deploy/model
```

Reads CSV or parquet in chunks (`--chunk-size`, default 50,000), so file size is limited by disk, not memory. Rows that
fail the range and allowed-value checks are set aside in `<output>_rejects.csv` (row number, loan id, column, check,
offending value) and the remaining rows are scored. A missing feature column aborts the run. The output file appears
only when the run finishes, so a half-written file is never mistaken for a result. The exit code is 2 if more than
`--max-reject-rate` (default 10%) of rows were rejected.

`<output>_summary.json` holds row counts, reject causes, the input file's SHA-256, the model version and scorer
checksum, decision shares, PD quantiles, throughput and, when the input has a `target` column, the observed default
rate in each decision band. Run it on the test split to see what each cutoff would have done:

```
py -c "import pandas as pd; d=pd.read_parquet('data/processed/model_table.parquet'); s=pd.read_parquet('artifacts/full/split.parquet').set_index('loan_id').split; d[d.loan_id.map(s)=='test'].to_parquet('test_loans.parquet')"
py -m creditrisk.batch --input test_loans.parquet --output scores_test.parquet
```

## 5. Load test

```
py -m creditrisk.loadtest --url http://127.0.0.1:8000 --requests 1000 --concurrency 1 4 16
```

Sends real loans sampled from the modeling table at each concurrency level and reports requests per second and latency
mean, p50, p90, p95, p99 and max, written to `reports/loadtest.json`. The client runs on the same machine as the
server, so the numbers exclude network time and reflect that machine only.

Reference run in the development sandbox (2 shared CPU cores, one worker, scorecard model, the load generator on the
same cores; yours will differ): a single scoring call takes about 8.6 ms in process; over HTTP the median is about
12 ms with one client at about 75 requests per second, and throughput stays near 75 to 80 requests per second at
higher concurrency while latency grows with the queue (p50 about 47 ms at 4 clients, about 196 ms at 16). The endpoint
is CPU-bound, so capacity comes from more worker processes, not more clients.

## Limits

- No authentication or rate limiting. Run it behind a gateway that provides both.
- The model and its cutoffs are fixed per image; there is no hot reload or A/B routing.
- The decision cutoffs are illustrative. Reason codes name model features, including `state`, which my earlier checks
  found acts largely as a proxy for disaster exposure; review whether state may appear in customer-facing reasons.
- The Docker image was written and its install steps were exercised outside Docker, but the sandbox this was built in
  has no Docker daemon, so `docker build` itself has not been run by the author.
