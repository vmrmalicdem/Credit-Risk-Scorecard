"""Score new loans with a saved model.

Usage (from repo root):
  py -m creditrisk.score --input loans.parquet --output scores.csv [--features full] [--model lgbm]
The input is a CSV or parquet file with one row per loan and the raw feature columns (the same columns as
data/processed/model_table.parquet). The columns and values are validated first (configs/project.yaml, validation),
then the saved model, which contains every preprocessing step, is applied unchanged. Use --model-uri to score with a
registered version instead of the local file, for example models:/creditrisk-full-lgbm@production.
Output: loan_id (if present), the calibrated default probability, and the scorecard points.
"""
import argparse

import joblib
import mlflow
import pandas as pd

from creditrisk.config import FEATURE_SETS, common_args, load_config, project_paths
from creditrisk.scoring import pd_to_score, points_scale
from creditrisk.validation import validate_scoring_input


def read_loans(path):
    return pd.read_csv(path) if str(path).lower().endswith(".csv") else pd.read_parquet(path)


def score_loans(df, scorer, cfg):
    """Validate df against the model's feature list and return calibrated probabilities and points."""
    validate_scoring_input(df, cfg, scorer.spec)
    pd_hat = scorer.predict_proba(df)[:, 1]
    factor, offset = points_scale(**{k: cfg["points"][k] for k in ("base_score", "base_odds", "pdo")})
    out = pd.DataFrame({"pd": pd_hat, "score": pd_to_score(pd_hat, factor, offset)}, index=df.index)
    if "loan_id" in df.columns:
        out.insert(0, "loan_id", df["loan_id"])
    return out


def main(argv=None):
    ap = common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--features", choices=FEATURE_SETS, default="full")
    ap.add_argument("--model", choices=["scorecard", "lgbm"], default="lgbm")
    ap.add_argument("--model-uri", help="registry model, e.g. models:/creditrisk-full-lgbm@production")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.model_uri:
        from creditrisk.tracking import setup

        setup(cfg)
        scorer = mlflow.sklearn.load_model(args.model_uri)
    else:
        scorer = joblib.load(project_paths(cfg)["artifacts"] / args.features / f"{args.model}.joblib")
    out = score_loans(read_loans(args.input), scorer, cfg)
    out.to_csv(args.output, index=False)
    source = args.model_uri or f"the {args.features} {args.model} model"
    print(f"Scored {len(out):,} loans with {source} -> {args.output}")


if __name__ == "__main__":
    main()
