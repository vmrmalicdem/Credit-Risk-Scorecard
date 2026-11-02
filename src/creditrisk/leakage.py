"""Leakage guard: checks the configured feature sets against the schema's known_at tags.

known_at = origination: fixed when the loan was made, safe as a feature.
known_at = performance: changes monthly, allowed only as a declared behaviour feature observed at or before the
prediction month (its source fields are listed in configs/project.yaml under features.behaviour).
known_at = outcome: populated at or after a termination or loss event, never allowed anywhere in the feature set.
"""

RESERVED_NAMES = {"target", "ead", "loan_id", "pred_period"}


class LeakageError(AssertionError):
    pass


def schema_tags(schema):
    return {c["name"]: c["known_at"] for c in schema["origination"] + schema["performance"]}


def check_feature_config(cfg, schema):
    """Raise LeakageError if any configured feature could carry post-origination or outcome information."""
    tags = schema_tags(schema)
    perf_names = {c["name"] for c in schema["performance"]}
    f = cfg["features"]
    problems = []

    origination = f["origination"]["numeric"] + f["origination"]["categorical"]
    behaviour = {**f["behaviour"]["numeric"], **f["behaviour"]["categorical"]}

    for name in origination:
        if tags.get(name) != "origination":
            problems.append(f"origination feature {name!r} is tagged {tags.get(name)!r}, expected 'origination'")
    for name, sources in behaviour.items():
        if not sources:
            problems.append(f"behaviour feature {name!r} declares no source fields")
        for src in sources:
            if src not in perf_names:
                problems.append(f"behaviour feature {name!r} uses unknown source {src!r}")
            elif tags[src] not in ("performance", "key"):
                problems.append(f"behaviour feature {name!r} uses {src!r} tagged {tags[src]!r}")
    for name in origination + list(behaviour):
        if name in RESERVED_NAMES or name.startswith("meta_") or tags.get(name) == "outcome":
            problems.append(f"feature {name!r} is an outcome, identifier or reporting-only column")
    dupes = {n for n in origination + list(behaviour) if (origination + list(behaviour)).count(n) > 1}
    if dupes:
        problems.append(f"duplicate feature names: {sorted(dupes)}")
    if problems:
        raise LeakageError("; ".join(problems))


def check_model_features(features, schema):
    """Raise LeakageError if a concrete list of model feature names includes an outcome or reserved column."""
    tags = schema_tags(schema)
    bad = [n for n in features if n in RESERVED_NAMES or n.startswith("meta_") or tags.get(n) == "outcome"]
    if bad:
        raise LeakageError(f"outcome or reserved columns used as features: {bad}")
