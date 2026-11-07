"""Schema, range and null-rate checks with Pandera, driven by configs/project.yaml (validation).

Three entry points share one rule set:
  validate_origination  the cleaned origination table (after sentinel codes become missing, before modeling)
  validate_model_table  the modeling table: one row per loan, binary target, positive EAD, feature ranges
  validate_scoring_input  feature columns of loans to be scored (value checks only; missing values are allowed
                          where the rule permits any, because null rates are a property of a batch, not a loan)
Each check returns a profile table (observed null rate, min, max, allowed values versus the limits) and raises
DataValidationError listing every failure when validation.on_failure is "raise", or warns when it is "warn".
"""
import warnings

import pandas as pd
import pandera.pandas as pa

from creditrisk.config import feature_lists


class DataValidationError(ValueError):
    pass


def _checks(rule):
    out = []
    if "min" in rule:
        out.append(pa.Check.ge(rule["min"], name=f"min {rule['min']}"))
    if "max" in rule:
        out.append(pa.Check.le(rule["max"], name=f"max {rule['max']}"))
    if "allowed" in rule:
        out.append(pa.Check.isin(rule["allowed"], name=f"allowed {rule['allowed']}"))
    if "pattern" in rule:
        out.append(pa.Check.str_matches(rule["pattern"], name=f"pattern {rule['pattern']}"))
    return out


def _null_rate_check(col, limit):
    return pa.Check(lambda df: bool(df[col].isna().mean() <= limit), name=f"null rate of {col} <= {limit}",
                    error=f"null rate of {col} above {limit}")


def build_schema(cfg, columns, table=None, null_rates=True):
    """Pandera schema for the given columns using the configured rules; table adds table-level rules."""
    vcfg = cfg["validation"]
    rules = vcfg["columns"]
    unknown = [c for c in columns if c not in rules]
    if unknown:
        raise KeyError(f"no validation rule configured for columns: {unknown}")
    cols, wide = {}, []
    for c in columns:
        limit = rules[c].get("max_null_rate", 0.0)
        cols[c] = pa.Column(checks=_checks(rules[c]), nullable=limit > 0, required=True)
        if null_rates and 0 < limit < 1:
            wide.append(_null_rate_check(c, limit))
    unique = None
    if table:
        t = vcfg["tables"][table]
        wide.append(pa.Check(lambda df: len(df) >= t["min_rows"], name=f"at least {t['min_rows']} rows",
                             error="too few rows"))
        if "min_defaults" in t:
            wide.append(pa.Check(lambda df: int(df["target"].sum()) >= t["min_defaults"],
                                 name=f"at least {t['min_defaults']} defaults", error="too few defaults"))
        if "default_rate" in t:
            lo, hi = t["default_rate"]
            wide.append(pa.Check(lambda df: lo <= df["target"].mean() <= hi, name=f"default rate in [{lo}, {hi}]",
                                 error="default rate outside the plausible range"))
        unique = t.get("unique")
    return pa.DataFrameSchema(cols, checks=wide, unique=unique, strict=False)


def profile(df, cfg, columns):
    """Observed null rate, range and levels next to the configured limits, for every checked column."""
    rules = cfg["validation"]["columns"]
    rows = []
    for c in columns:
        s, r = df[c], rules[c]
        numeric = pd.api.types.is_numeric_dtype(s) and "allowed" not in r
        rows.append({
            "column": c, "null_rate": float(s.isna().mean()), "max_null_rate": r.get("max_null_rate", 0.0),
            "observed_min": float(s.min()) if numeric and s.notna().any() else None,
            "observed_max": float(s.max()) if numeric and s.notna().any() else None,
            "limit_min": r.get("min"), "limit_max": r.get("max"),
            "observed_levels": ",".join(sorted(map(str, s.dropna().unique())))[:80] if not numeric else None,
            "allowed": ",".join(map(str, r["allowed"])) if "allowed" in r else r.get("pattern"),
        })
    return pd.DataFrame(rows)


def _run(df, cfg, columns, table, null_rates, label):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"{label}: missing columns {missing}")
    df = df.reset_index(drop=True)
    schema = build_schema(cfg, columns, table, null_rates)
    prof = profile(df, cfg, columns)
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        failures = err.failure_cases.assign(table=label)
        cols = ["table", "column", "check", "failure_case", "index"]
        failures = failures[[c for c in cols if c in failures.columns]]
        message = f"{label}: {len(failures)} validation failure(s):\n" + failures.head(25).to_string(index=False)
        if cfg["validation"]["on_failure"] == "raise":
            raise DataValidationError(message) from None
        warnings.warn(message, stacklevel=3)
        return prof, failures
    return prof, pd.DataFrame(columns=["table", "column", "check", "failure_case", "index"])


def validate_origination(df, cfg):
    cols = [c for c in cfg["validation"]["columns"] if c in df.columns and c in
            set(cfg["features"]["origination"]["numeric"] + cfg["features"]["origination"]["categorical"])]
    return _run(df, cfg, ["loan_id"] + cols, "origination", True, "origination")


def validate_model_table(df, cfg):
    spec = feature_lists(cfg, "full")
    cols = ["loan_id", "pred_period", "target", "ead", "meta_disaster_event", "meta_assist_event"]
    return _run(df, cfg, cols + spec["numeric"] + spec["categorical"], "model_table", True, "model_table")


def validate_scoring_input(df, cfg, spec):
    cols = spec["numeric"] + spec["categorical"]
    return _run(df, cfg, cols, None, False, "scoring_input")


def invalid_rows(df, cfg, spec):
    """Row-level scoring-input check for batches: (positions of failing rows, one failure record per problem).

    Missing columns still raise DataValidationError, because no row can be scored without them.
    """
    cols = spec["numeric"] + spec["categorical"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataValidationError(f"scoring_input: missing columns {missing}")
    try:
        build_schema(cfg, cols, None, False).validate(df.reset_index(drop=True), lazy=True)
    except pa.errors.SchemaErrors as err:
        f = err.failure_cases
        f = f[f["index"].notna()]
        out = pd.DataFrame({"row": f["index"].astype(int), "column": f["column"], "check": f["check"].astype(str),
                            "failure_case": f["failure_case"].astype(str)})
        return sorted(out.row.unique()), out
    return [], pd.DataFrame(columns=["row", "column", "check", "failure_case"])
