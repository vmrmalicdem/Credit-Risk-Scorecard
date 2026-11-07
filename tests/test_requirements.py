"""Reproducibility: every dependency in requirements.txt is pinned to an exact version."""
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = re.compile(r"^[A-Za-z0-9_.\-]+(\[[A-Za-z0-9_,\-]+\])?==[A-Za-z0-9_.\-+!]+$")


def pinned_lines():
    lines = (ROOT / "requirements.txt").read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def name(line):
    return re.split(r"[=\[]", line)[0].lower().replace("_", "-")


def test_every_requirement_is_pinned_with_double_equals():
    unpinned = [ln for ln in pinned_lines() if not PIN.match(ln)]
    assert not unpinned, f"not pinned to an exact version: {unpinned}"


def test_every_direct_dependency_is_in_the_pin_file():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    direct = {re.split(r"[<>=!\[ ]", d)[0].lower().replace("_", "-") for d in project["dependencies"]}
    direct |= {re.split(r"[<>=!\[ ]", d)[0].lower().replace("_", "-") for d in project["optional-dependencies"]["dev"]}
    assert direct <= {name(ln) for ln in pinned_lines()}


def test_no_package_is_pinned_twice():
    names = [name(ln) for ln in pinned_lines()]
    assert len(names) == len(set(names))


def serve_pins():
    lines = (ROOT / "requirements-serve.txt").read_text().splitlines()
    return {name(ln.strip()): ln.strip().split("==")[1] for ln in lines if ln.strip() and not ln.startswith("#")}


def test_serve_pins_are_exact_and_agree_with_the_full_pins():
    serve = serve_pins()
    full = {name(ln): ln.split("==")[1] for ln in pinned_lines()}
    assert {"fastapi", "uvicorn", "pandas", "scikit-learn", "lightgbm", "pandera"} <= set(serve)
    assert not {"mlflow", "matplotlib", "pytest", "ruff"} & set(serve)
    assert {k: (v, full[k]) for k, v in serve.items() if full.get(k) != v} == {}


def test_every_serving_import_is_covered_by_the_serve_pins():
    serve = set(serve_pins())
    needed = {"numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "pyyaml", "joblib", "pyarrow", "pandera",
              "fastapi", "uvicorn", "pydantic"}
    assert needed <= serve
