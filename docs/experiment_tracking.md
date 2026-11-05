# Experiment tracking and model registry

Every call to `py -m creditrisk.train` is logged to MLflow, and each fitted model is registered as a new version.
The store is local (a SQLite file and an artifact folder under `mlruns/`, git-ignored because it holds models fitted
on loan data). Settings are in `configs/project.yaml` under `mlflow`.

## What is logged

| Where | What |
|---|---|
| Parameters | seed, split, scorecard and LightGBM settings, the values chosen by cross-validation (`scorecard.chosen.C`, `lgbm.chosen.*`), feature lists, row and default counts |
| Metrics | test metrics for each model, raw and Platt-calibrated, as `test.<model>_<variant>.<metric>` with `.ci_low` and `.ci_high` bootstrap bounds; cross-validated log loss as `cv.<model>.neg_log_loss` |
| Tags | git commit and branch, whether code had uncommitted changes, SHA-256 of the config file, fingerprint of the modeling table, SHA-256 of both raw files, SHA-256 of `requirements.txt` |
| Artifacts | `inputs/` (config file, data manifest), `lineage/lineage.json`, `lineage/code_uncommitted.diff` (only if the code was dirty), `reports/`, `run/run_summary.json`, `run/split.parquet`, and each model with its signature |

Two models are registered per feature set: `creditrisk-full-scorecard`, `creditrisk-full-lgbm`,
`creditrisk-origination-scorecard` and `creditrisk-origination-lgbm`. A registered model is the whole
`CalibratedScorer` (raw-input pipeline plus Platt recalibration), so scoring needs no separate preprocessing.

## Labels

Versions are labelled with registry **aliases**, MLflow's replacement for the deprecated Staging/Production stages.

- `candidate`: set automatically on the newest version after every training run.
- `production`: set only by `registry promote`, after the gates below pass.
- `previous_production`: the version that `production` replaced, kept so `registry rollback` can restore it.

## Commands

```
py -m creditrisk.registry list
py -m creditrisk.registry lineage --name creditrisk-full-lgbm --alias candidate --check
py -m creditrisk.registry promote --name creditrisk-full-lgbm --version 1
py -m creditrisk.registry rollback --name creditrisk-full-lgbm
py -m creditrisk.score --input loans.parquet --output scores.csv --model-uri models:/creditrisk-full-lgbm@production
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

## Which data, code and parameters produced this model?

`registry lineage --name <model> --version <n>` (or `--alias production`) prints, for that version:

- **Data:** the raw origination and performance files with their SHA-256 hashes, the modeling table's row count,
  default count and content fingerprint, and whether the table matched its manifest when training started
  (`build_dataset` writes `data/processed/data_manifest.json`).
- **Code:** the git commit and branch, and whether `src/`, `configs/`, `pyproject.toml` or `requirements.txt` had
  uncommitted changes. If they did, the diff is stored with the run, so even an uncommitted model can be traced.
- **Parameters:** the configuration hash (the full file is stored), seed and split, and the values chosen by tuning.
- **Environment:** Python and package versions and the hash of `requirements.txt`.
- **Results:** the test metrics with their 95% intervals.

Add `--check` to compare all of that with the current checkout. Each line reads `MATCH` or `DIFFERS`, so you can tell
whether the model could be rebuilt from what is on disk now. To rebuild an old version: check out its commit, put the
raw files with the recorded hashes in `data/raw`, and rerun the pipeline.

## Promotion gates

`registry promote` refuses to label a version `production` unless all of these hold (thresholds are in the config):

1. test ROC-AUC is at least `min_test_roc_auc`;
2. the code was clean when it was trained (`require_clean_code`);
3. the modeling table matched its data manifest (`require_matching_data`);
4. test ROC-AUC is no more than `max_roc_auc_drop_vs_production` below the current production version.

`--force --reason "..."` overrides the gates. The reason and the failed gates are stored on the version as a tag,
together with who promoted it and when. The gates are guardrails, not a validation of the model: with 66 test defaults
the AUC interval is wide, so a human still decides.

## Limits

- The store is local to one machine. Sharing it needs an MLflow tracking server and remote artifact storage.
- A fingerprint identifies the data and a hash identifies a file; neither can rebuild the raw files themselves.
- MLflow usage telemetry is switched off in `creditrisk/__init__.py`.
