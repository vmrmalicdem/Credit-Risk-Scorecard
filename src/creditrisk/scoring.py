"""Points-scaled scorecard: turns logistic coefficients into additive bin points.

score = offset + factor x ln(good:bad odds), with factor = pdo / ln 2 and offset = base_score - factor x ln(base_odds).
Higher score = lower risk. A loan's score is base_points + sum of the points of the bins it falls in, which equals
offset - factor x (the model's log-odds of default), so the points table is an exact restatement of the model.
"""
import numpy as np
import pandas as pd

from creditrisk.features import scorecard_frame


def points_scale(base_score, base_odds, pdo):
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return factor, offset


def coefficients_to_points(coefs, intercept, factor, offset):
    """Return (points per bin, base points) from logistic coefficients and intercept (log-odds of default)."""
    return -factor * np.asarray(coefs), offset - factor * intercept


def score_from_design(design, bin_points, base_points):
    """Score for each row of a binned (one-hot) design matrix."""
    return np.asarray(design @ bin_points).ravel() + base_points


def pd_to_score(pd_prob, factor, offset):
    """Score implied directly by a default probability: offset - factor x logit(PD)."""
    p = np.asarray(pd_prob, dtype=float)
    return offset - factor * np.log(p / (1 - p))


def numeric_labels(binner):
    """Human-readable bin labels per numeric feature, in the binner's column order."""
    out = {}
    for c in binner.columns_:
        e = binner.edges_[c]
        labs = [f"<= {e[0]:g}"] if len(e) else ["all"]
        labs += [f"> {e[k - 1]:g} to {e[k]:g}" for k in range(1, len(e))]
        if len(e):
            labs.append(f"> {e[-1]:g}")
        if binner.has_na_[c]:
            labs.append("missing")
        out[c] = labs
    return out


def feature_bins(pre, spec):
    """(feature, bin label) for every column of the fitted preprocessor, in column order."""
    labs = numeric_labels(pre.named_transformers_["num"])
    cats = sorted(spec["categorical"], key=len, reverse=True)
    rows = []
    for name in pre.get_feature_names_out():
        kind, rest = name.split("__", 1)
        if kind == "num":
            feat, tail = rest.rsplit("__", 1)
            k = len(labs[feat]) - 1 if tail == "missing" else int(tail.replace("bin", ""))
            rows.append((feat, labs[feat][k]))
        else:
            feat = next(c for c in cats if rest.startswith(c + "_"))
            level = rest[len(feat) + 1:]
            rows.append((feat, "other (rare levels)" if level == "infrequent_sklearn" else level))
    return rows


def scorecard_table(pipeline, spec, dev, scale):
    """Points table with development-set loan counts and default rates; first row is the base points."""
    pre, lr = pipeline.named_steps["pre"], pipeline.named_steps["lr"]
    factor, offset = scale
    design = pre.transform(scorecard_frame(dev, spec))
    y = dev.target.to_numpy()
    counts = np.asarray(design.sum(axis=0)).ravel()
    events = np.asarray(design.T.dot(y)).ravel()
    bins = feature_bins(pre, spec)
    pts, base = coefficients_to_points(lr.coef_[0], lr.intercept_[0], factor, offset)
    t = pd.DataFrame({"feature": [b[0] for b in bins], "bin": [b[1] for b in bins],
                      "coefficient": lr.coef_[0], "points": pts,
                      "loans_dev": counts.astype(int), "defaults_dev": events.astype(int)})
    t["default_rate_dev"] = t.defaults_dev / t.loans_dev.where(t.loans_dev > 0)
    first = pd.DataFrame([{"feature": "BASE", "bin": "intercept", "coefficient": lr.intercept_[0],
                           "points": base, "loans_dev": len(dev), "defaults_dev": int(y.sum()),
                           "default_rate_dev": y.mean()}])
    return pd.concat([first, t], ignore_index=True)


def points_range_by_feature(table, min_loans):
    """Spread of points between the lowest and highest bin of each feature, ignoring sparse bins."""
    body = table[(table.feature != "BASE") & (table.loans_dev >= min_loans)]
    return body.groupby("feature").points.agg(lambda s: s.max() - s.min()).sort_values(ascending=False)
