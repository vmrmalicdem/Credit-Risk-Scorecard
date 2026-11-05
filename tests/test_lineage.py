import shutil
import subprocess

import pandas as pd
import pytest

from creditrisk.lineage import git_info, sha256_file, sha256_json, table_fingerprint

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def git(root, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=root, check=True,
                   capture_output=True)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "notes.txt").write_text("outside the code paths\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_table_fingerprint_is_stable_and_sensitive_to_values_columns_and_order():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", None]})
    assert table_fingerprint(df) == table_fingerprint(df.copy())
    assert table_fingerprint(df) != table_fingerprint(df.assign(a=[1, 2, 4]))
    assert table_fingerprint(df) != table_fingerprint(df.rename(columns={"a": "c"}))
    assert table_fingerprint(df) != table_fingerprint(df.iloc[::-1])


def test_file_and_json_hashes(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("abc")
    assert sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})


def test_outside_a_git_repository_everything_is_unknown(tmp_path):
    info = git_info(tmp_path, ["src"])
    assert info["commit"] == info["dirty"] == "unknown" and info["dirty_files"] == []


@needs_git
def test_clean_repo_reports_commit_and_not_dirty(repo):
    info = git_info(repo, ["src"])
    assert len(info["commit"]) == 40 and info["dirty"] == "false" and info["diff"] == ""


@needs_git
def test_edit_in_a_code_path_marks_dirty_and_keeps_the_diff(repo):
    (repo / "src" / "a.py").write_text("x = 2\n")
    info = git_info(repo, ["src"])
    assert info["dirty"] == "true" and "src/a.py" in info["dirty_files"] and "+x = 2" in info["diff"]


@needs_git
def test_new_untracked_code_file_marks_dirty(repo):
    (repo / "src" / "b.py").write_text("y = 1\n")
    assert git_info(repo, ["src"])["dirty"] == "true"


@needs_git
def test_changes_outside_the_code_paths_do_not_mark_dirty(repo):
    (repo / "notes.txt").write_text("edited\n")
    assert git_info(repo, ["src"])["dirty"] == "false"
