"""Configuration loading and feature-set helpers shared by every pipeline step."""
from pathlib import Path

import yaml

DEFAULT_CONFIG = "configs/project.yaml"
DEFAULT_SCHEMA = "configs/schema_map.yaml"
FEATURE_SETS = ("full", "origination")


def load_yaml(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_config(path=DEFAULT_CONFIG):
    return load_yaml(path)


def load_schema(path=DEFAULT_SCHEMA):
    return load_yaml(path)


def project_paths(cfg):
    return {k: Path(v) for k, v in cfg["paths"].items()}


def feature_lists(cfg, feature_set):
    """Numeric and categorical feature names for 'full' (origination + behaviour) or 'origination'."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}")
    f = cfg["features"]
    numeric, categorical = list(f["origination"]["numeric"]), list(f["origination"]["categorical"])
    if feature_set == "full":
        numeric += list(f["behaviour"]["numeric"])
        categorical += list(f["behaviour"]["categorical"])
    return {"numeric": numeric, "categorical": categorical}


def common_args(parser):
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser
