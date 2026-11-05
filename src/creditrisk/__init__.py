"""Interpretable credit-risk scorecard and stress-tested expected loss on one mortgage vintage."""

__version__ = "0.1.0"

import os

# MLflow sends anonymous usage telemetry by default; this project handles loan data, so keep it off.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
