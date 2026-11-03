"""Shared feature helpers: quantile binner for the scorecard, frame builders and the pipeline steps that use them.

ScorecardFrame and LgbmFrame are the first step of each model's Pipeline, so a raw loan table goes through exactly
the same selection and cleaning at training and at scoring time. Saved models are pickled with a reference to this
module's path (creditrisk.features); do not rename it without retraining.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class QuantileBinner(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=5):
        self.n_bins = n_bins

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.edges_, self.has_na_ = {}, {}
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        for c in self.columns_:
            v = pd.to_numeric(X[c], errors="coerce")
            self.has_na_[c] = bool(v.isna().any())
            self.edges_[c] = np.unique(np.quantile(v.dropna(), qs)) if v.notna().any() else np.array([])
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        blocks = []
        for c in self.columns_:
            v = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float)
            e = self.edges_[c]
            width = len(e) + 1 + int(self.has_na_[c])
            b = np.searchsorted(e, v, side="left")
            nan = np.isnan(v)
            b = np.where(nan, len(e) + 1 if self.has_na_[c] else -1, b)
            m = np.zeros((len(v), width))
            ok = b >= 0
            m[np.flatnonzero(ok), b[ok]] = 1.0
            blocks.append(m)
        return np.hstack(blocks)

    def get_feature_names_out(self, input_features=None):
        names = []
        for c in self.columns_:
            names += [f"{c}__bin{k}" for k in range(len(self.edges_[c]) + 1)]
            if self.has_na_[c]:
                names.append(f"{c}__missing")
        return np.array(names, dtype=object)


def scorecard_frame(df, spec):
    X = df[spec["numeric"] + spec["categorical"]].copy()
    for c in spec["categorical"]:
        X[c] = X[c].astype(object).where(X[c].notna(), "MISSING").astype(str)
    return X


def category_levels(df, spec):
    return {c: sorted(df[c].dropna().astype(str).unique()) for c in spec["categorical"]}


def lgbm_frame(df, spec, levels):
    X = df[spec["numeric"] + spec["categorical"]].copy()
    for c in spec["categorical"]:
        values = X[c].astype("object")
        X[c] = pd.Categorical(values.where(values.isin(levels[c]), None), categories=levels[c])
    return X


def _require_columns(df, spec):
    missing = [c for c in spec["numeric"] + spec["categorical"] if c not in df.columns]
    if missing:
        raise ValueError(f"input is missing feature columns: {missing}")


class ScorecardFrame(BaseEstimator, TransformerMixin):
    """Select the scorecard's feature columns and mark missing categories; stateless, so nothing is learned."""

    def __init__(self, numeric, categorical):
        self.numeric = numeric
        self.categorical = categorical

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):
        return True

    def transform(self, X):
        spec = {"numeric": self.numeric, "categorical": self.categorical}
        _require_columns(X, spec)
        return scorecard_frame(X, spec)


class LgbmFrame(BaseEstimator, TransformerMixin):
    """Select feature columns and give categoricals a fixed vocabulary; unseen values become missing."""

    def __init__(self, numeric, categorical, levels):
        self.numeric = numeric
        self.categorical = categorical
        self.levels = levels

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):
        return True

    def transform(self, X):
        spec = {"numeric": self.numeric, "categorical": self.categorical}
        _require_columns(X, spec)
        return lgbm_frame(X, spec, self.levels)
