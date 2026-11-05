"""Model pipelines and the calibrated scorer that is saved to disk.

Each model is ONE scikit-learn Pipeline that starts from the raw loan table: column selection and cleaning,
preprocessing (binning and one-hot encoding for the scorecard), then the estimator. Training, cross-validation,
recalibration, stress testing and scoring all call the same object, so the transformations cannot drift apart
between training and serving. CalibratedScorer adds the Platt recalibration on top and is what train.py saves.
"""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from creditrisk.features import LgbmFrame, QuantileBinner, ScorecardFrame
from creditrisk.metrics import apply_platt


def build_scorecard_pipeline(spec, cfg):
    pre = ColumnTransformer([
        ("num", QuantileBinner(cfg["scorecard"]["n_bins"]), spec["numeric"]),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=cfg["scorecard"]["min_category_frequency"]), spec["categorical"]),
    ])
    return Pipeline([("frame", ScorecardFrame(spec["numeric"], spec["categorical"])), ("pre", pre),
                     ("lr", LogisticRegression(max_iter=3000))])


def build_lgbm_pipeline(spec, cfg, levels, params=None):
    model = LGBMClassifier(**cfg["lgbm"]["fixed"], **(params or {}), random_state=cfg["seed"], n_jobs=1,
                           verbose=-1, deterministic=True, force_row_wise=True)
    return Pipeline([("frame", LgbmFrame(spec["numeric"], spec["categorical"], levels)), ("model", model)])


class CalibratedScorer(BaseEstimator, ClassifierMixin):
    """A fitted raw-input pipeline plus Platt recalibration; predict_proba returns calibrated default probabilities."""

    def __init__(self, pipeline, platt, eps):
        self.pipeline = pipeline
        self.platt = platt
        self.eps = eps
        self.classes_ = np.array([0, 1])

    def __sklearn_is_fitted__(self):
        return True

    @property
    def spec(self):
        frame = self.pipeline.named_steps["frame"]
        return {"numeric": list(frame.numeric), "categorical": list(frame.categorical)}

    def predict_raw(self, X):
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X):
        """Calibrated default probability, one value per loan (what a registry or serving layer returns)."""
        return self.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        q = apply_platt(self.predict_raw(X), self.platt, self.eps)
        return np.column_stack([1 - q, q])
