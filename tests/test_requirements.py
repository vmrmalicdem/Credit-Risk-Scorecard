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
