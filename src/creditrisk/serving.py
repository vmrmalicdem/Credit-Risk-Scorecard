"""Scoring service shared by the API and the batch script, so both produce identical results.

For each loan it returns the calibrated probability of default, the scorecard-scale score, a decision band and the
reasons that cost the most risk.

Reason impacts:
  scorecard  points lost against the best bin of each feature (the classic "points below maximum" method); a value
             the model has no bin for (unseen or missing without a missing bin) contributes zero and is never penalised
  lgbm       positive TreeSHAP contributions (log-odds of default pushed up)
Impacts are on the model's raw scale, before Platt recalibration, which changes probabilities only slightly.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from creditrisk.config import load_config
from creditrisk.leakage import RESERVED_NAMES
from creditrisk.scoring import coefficients_to_points, feature_bins, pd_to_score, points_scale

METADATA = "metadata.json"
SCORER = "scorer.joblib"
CONFIG = "project.yaml"


class ModelLoadError(RuntimeError):
    pass


def sha256_bytes_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_kind(scorer):
    last = scorer.pipeline.steps[-1][1]
    return "lgbm" if hasattr(last, "booster_") else "scorecard"


def decide(pd_hat, lower, upper, labels):
    """Decision band for each probability: labels[0] below lower, labels[2] at or above upper, else labels[1]."""
    p = np.asarray(pd_hat, dtype=float)
    return np.where(p < lower, labels[0], np.where(p >= upper, labels[2], labels[1]))


def top_reasons(impact, features, k):
    """Per row, up to k (feature, impact) pairs with positive impact, largest first."""
    order = np.argsort(-impact, axis=1, kind="stable")[:, :k]
    out = []
    for row, idx in zip(impact, order, strict=True):
        out.append([(features[j], float(row[j])) for j in idx if row[j] > 1e-9])
    return out


class ScorecardImpact:
    def __init__(self, scorer, scale):
        pipe = scorer.pipeline
        pre, lr = pipe.named_steps["pre"], pipe.named_steps["lr"]
        factor, offset = scale
        points, _ = coefficients_to_points(lr.coef_[0], lr.intercept_[0], factor, offset)
        self.features = scorer.spec["numeric"] + scorer.spec["categorical"]
        owner = [b[0] for b in feature_bins(pre, scorer.spec)]
        self.groups = {f: np.flatnonzero(np.array(owner) == f) for f in self.features}
        self.points = points
        self.best = {f: points[idx].max() for f, idx in self.groups.items()}
        self.pipe = pipe

    def __call__(self, design):
        design = design.toarray() if hasattr(design, "toarray") else np.asarray(design)
        got = design * self.points
        out = np.zeros((len(design), len(self.features)))
        for j, f in enumerate(self.features):
            out[:, j] = self.best[f] - got[:, self.groups[f]].sum(axis=1)
        return np.maximum(out, 0.0)


class LgbmImpact:
    def __init__(self, scorer):
        self.pipe = scorer.pipeline
        self.features = scorer.spec["numeric"] + scorer.spec["categorical"]
        self.booster = self.pipe.named_steps["model"].booster_

    def __call__(self, X):
        contrib = self.booster.predict(X, pred_contrib=True)[:, :-1]
        return np.maximum(contrib, 0.0)


class ScoringService:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        try:
            self.metadata = json.loads((model_dir / METADATA).read_text())
            self.cfg = load_config(model_dir / CONFIG)
        except OSError as err:
            raise ModelLoadError(f"model directory {model_dir} is incomplete: {err}") from err
        actual = sha256_bytes_of(model_dir / SCORER)
        if actual != self.metadata["scorer_sha256"]:
            raise ModelLoadError("scorer.joblib does not match the checksum in metadata.json; re-export the model")
        self.scorer = joblib.load(model_dir / SCORER)
        self.kind = model_kind(self.scorer)
        self.spec = self.scorer.spec
        self.features = self.spec["numeric"] + self.spec["categorical"]
        bad = [f for f in self.features if f in RESERVED_NAMES or f.startswith("meta_")]
        if bad:
            raise ModelLoadError(f"model uses outcome or reserved columns: {bad}")
        self.feature_set = self.metadata["feature_set"]
        s = self.cfg["serving"]
        self.max_reasons = s["max_reasons"]
        self.lower, self.upper = s["decision"]["lower_pd"], s["decision"]["upper_pd"]
        self.labels = s["decision"]["labels"][self.feature_set]
        self.codes = s["reason_codes"]
        missing = [f for f in self.features if f not in self.codes]
        if missing:
            raise ModelLoadError(f"no reason code configured for features: {missing}")
        self.scale = points_scale(**{k: self.cfg["points"][k] for k in ("base_score", "base_odds", "pdo")})
        self.impact = (LgbmImpact(self.scorer) if self.kind == "lgbm"
                       else ScorecardImpact(self.scorer, self.scale))

    def score(self, df, with_reasons=True):
        """Scores for every row of df (raw loan columns): pd, score, decision and optionally the reasons."""
        df = df.reset_index(drop=True)
        pipe = self.scorer.pipeline
        design = pipe[:-1].transform(df)
        p = self.scorer.calibrate(pipe.steps[-1][1].predict_proba(design)[:, 1])[:, 1]
        out = pd.DataFrame({"pd": p, "score": np.rint(pd_to_score(p, *self.scale)).astype(int),
                            "decision": decide(p, self.lower, self.upper, self.labels)})
        if with_reasons:
            out["reasons"] = top_reasons(self.impact(design), self.impact.features, self.max_reasons)
        return out

    def reason_dicts(self, reasons, row):
        return [{"rank": i + 1, "code": self.codes[f]["code"], "feature": f, "description": self.codes[f]["text"],
                 "value": _clean(row.get(f)), "impact": round(v, 4)} for i, (f, v) in enumerate(reasons)]

    def describe(self):
        m = self.metadata
        return {"name": m.get("registry_name"), "version": m.get("registry_version"), "kind": self.kind,
                "feature_set": self.feature_set, "decision_labels": self.labels,
                "decision_cutoffs_pd": {"lower": self.lower, "upper": self.upper}, "features": self.features,
                "lineage": {k: m.get(k) for k in ("run_id", "git_commit", "code_dirty", "data_fingerprint",
                                                  "config_sha256", "exported_at_utc", "test_roc_auc")}}


def _clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v.item() if hasattr(v, "item") else v
