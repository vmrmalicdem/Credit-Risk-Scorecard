import re
from pathlib import Path

import pytest

from creditrisk.config import project_paths

ROOT = Path(__file__).resolve().parents[1]


def test_required_sections_exist(cfg):
    for key in ["seed", "prediction_rule", "outcome_window_months", "default_event", "split", "paths", "features",
                "scorecard", "lgbm", "evaluation", "lgd", "stress", "points"]:
        assert key in cfg, key


def test_parameters_are_in_valid_ranges(cfg):
    assert 0 < cfg["split"]["test_size"] < 1 and cfg["split"]["cv_folds"] >= 2
    assert cfg["scorecard"]["n_bins"] >= 2 and all(c > 0 for c in cfg["scorecard"]["c_grid"])
    assert 0 < cfg["evaluation"]["top_share"] < 1 and cfg["evaluation"]["n_bootstrap"] >= 1
    assert cfg["lgd"]["floor"] <= cfg["lgd"]["cap"]
    assert cfg["points"]["pdo"] > 0 and cfg["points"]["base_odds"] > 0
    assert 0.5 < cfg["stress"]["reference_quantile"] < 1


def test_stress_scenarios_include_a_baseline_with_no_shock(cfg):
    base = cfg["stress"]["scenarios"]["baseline"]
    assert base["ltv_pp"] == 0 and base["rate_pts"] == 0


def test_expected_signs_refer_to_configured_features(cfg):
    f = cfg["features"]
    names = set(f["origination"]["numeric"]) | set(f["behaviour"]["numeric"])
    assert set(f["expected_signs"]) <= names


def test_project_paths_are_paths(cfg):
    assert all(isinstance(p, Path) for p in project_paths(cfg).values())


def test_source_code_does_not_hardcode_the_configured_parameters():
    """Guards against parameters creeping back into code: none of these literals may appear in the package."""
    banned = [r"\b0\.30\b", r"n_estimators\s*=\s*300", r"\b2000\b", r"LTV_CAP", r"\bN_BOOT\b", r"\bMIN_SEG\b"]
    for path in (ROOT / "src" / "creditrisk").glob("*.py"):
        text = path.read_text()
        for pattern in banned:
            assert not re.search(pattern, text), f"{path.name} contains {pattern}"


def test_config_file_loads_from_the_repo_root():
    assert (ROOT / "configs" / "project.yaml").exists() and (ROOT / "configs" / "schema_map.yaml").exists()


@pytest.mark.parametrize("module", ["build_dataset", "train", "ecl_stress", "interpret", "state_check", "report"])
def test_pipeline_modules_import_without_side_effects(module):
    __import__(f"creditrisk.{module}")
