"""Scoring API.

Run:   MODEL_DIR=deploy/model py -m creditrisk.api          (HOST, PORT, WORKERS, LOG_LEVEL are also read)
Docs:  http://127.0.0.1:8000/docs
POST /v1/score   one loan -> probability of default, score, decision and top reason codes
GET  /v1/model   which model is loaded and where it came from
GET  /health     the process is up            GET /ready   the model is loaded and has scored a warm-up row

The request schema is built from the model's feature list and the validation rules in the model folder's project.yaml
(the same ranges and allowed values the training data was checked against). Unknown fields are rejected, so a
misspelled field name can never be silently scored as a missing value. Logs carry request ids, latency, decision and
model version, never the applicant's field values.
"""
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model

from creditrisk.logs import log_config
from creditrisk.serving import ModelLoadError, ScoringService

log = logging.getLogger("creditrisk.api")
REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def request_model(service):
    """Pydantic model of one loan, built from the model's features and the configured validation rules."""
    rules = service.cfg["validation"]["columns"]
    fields = {"loan_id": (str | None, Field(None, max_length=64, description="Echoed back; not used for scoring"))}
    for name in service.features:
        rule = rules[name]
        kw = {"description": service.codes[name]["text"]}
        if "allowed" in rule:
            typ = Literal[tuple(rule["allowed"])]
        elif "pattern" in rule:
            typ, kw["pattern"] = str, rule["pattern"]
        else:
            typ, kw["allow_inf_nan"] = float, False
            kw.update({k: rule[k] for k in ("min", "max") if k in rule})
            kw = {{"min": "ge", "max": "le"}.get(k, k): v for k, v in kw.items()}
        if rule.get("max_null_rate", 0) > 0:
            fields[name] = (typ | None, Field(None, **kw))
        else:
            fields[name] = (typ, Field(..., **kw))
    return create_model("LoanApplication", __config__=ConfigDict(extra="forbid"), **fields)


class ReasonCode(BaseModel):
    rank: int
    code: str
    feature: str
    description: str
    value: float | str | None = None
    impact: float = Field(description="Points lost (scorecard) or log-odds added (LightGBM) versus the best case")


class ModelRef(BaseModel):
    name: str | None
    version: int | None
    kind: str
    feature_set: str


class ScoreResponse(BaseModel):
    request_id: str
    loan_id: str | None
    pd: float = Field(description="Calibrated probability of default in the 12 months after the prediction month")
    score: int = Field(description="Scorecard points; higher means lower risk")
    decision: str
    reason_codes: list[ReasonCode]
    model: ModelRef


def to_frame(application, service):
    row = application.model_dump(exclude={"loan_id"})
    df = pd.DataFrame([row])
    df[service.spec["numeric"]] = df[service.spec["numeric"]].astype(float)
    return df


def create_app(model_dir=None):
    started = time.perf_counter()
    service = ScoringService(model_dir or os.environ.get("MODEL_DIR", "deploy/model"))
    Application = request_model(service)

    @asynccontextmanager
    async def lifespan(app):
        blank = {f: (np.nan if f in service.spec["numeric"] else None) for f in service.features}
        service.score(pd.DataFrame([blank]))
        app.state.service = service
        log.info("model loaded", extra={"fields": {
            "event": "startup", "model": service.metadata.get("registry_name"),
            "version": service.metadata.get("registry_version"), "kind": service.kind,
            "feature_set": service.feature_set, "seconds": round(time.perf_counter() - started, 3)}})
        yield

    app = FastAPI(title="Credit risk scoring API", version="1.0", lifespan=lifespan)
    app.state.service = None

    @app.middleware("http")
    async def observe(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        request.state.request_id = incoming if REQUEST_ID.match(incoming) else uuid.uuid4().hex
        request.state.fields = {}
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error", extra={"fields": {"event": "error", "request_id": request.state.request_id,
                                                               "path": request.url.path}})
            response = JSONResponse({"detail": "internal error", "request_id": request.state.request_id}, 500)
        ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request.state.request_id
        if request.url.path not in ("/health", "/ready"):
            log.info("request", extra={"fields": {
                "event": "request", "request_id": request.state.request_id, "method": request.method,
                "path": request.url.path, "status": response.status_code, "latency_ms": round(ms, 2),
                "model_version": service.metadata.get("registry_version"),
                **request.state.fields}})
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        request.state.fields = {"validation_errors": [".".join(map(str, e["loc"][1:])) or "body" for e in errors]}
        return JSONResponse({"detail": errors, "request_id": request.state.request_id}, 422)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        if app.state.service is None:
            return JSONResponse({"status": "loading"}, 503)
        return {"status": "ready"}

    @app.get("/v1/model")
    def model_info():
        return service.describe()

    @app.post("/v1/score", response_model=ScoreResponse)
    async def score(application: Application, request: Request):
        row = application.model_dump(exclude={"loan_id"})
        scored = service.score(to_frame(application, service)).iloc[0]
        request.state.fields = {"decision": scored["decision"], "pd": round(float(scored["pd"]), 6),
                                "score": int(scored["score"])}
        return {"request_id": request.state.request_id, "loan_id": application.loan_id, "pd": float(scored["pd"]),
                "score": int(scored["score"]), "decision": scored["decision"],
                "reason_codes": service.reason_dicts(scored["reasons"], row),
                "model": {"name": service.metadata.get("registry_name"),
                          "version": service.metadata.get("registry_version"), "kind": service.kind,
                          "feature_set": service.feature_set}}

    return app


def main():
    port, workers = int(os.environ.get("PORT", 8000)), int(os.environ.get("WORKERS", 1))
    try:
        uvicorn.run("creditrisk.api:create_app", factory=True, host=os.environ.get("HOST", "127.0.0.1"), port=port,
                    workers=workers, log_config=log_config(os.environ.get("LOG_LEVEL", "INFO")), access_log=False)
    except ModelLoadError as err:
        raise SystemExit(f"cannot start: {err}") from err


if __name__ == "__main__":
    main()
