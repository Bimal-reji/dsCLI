"""CLI tests using Typer's CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from dscli.cli import app

runner = CliRunner()


def _write_project_config(root: Path) -> None:
    """Write a fast, deterministic config for CLI tests."""
    cfg = {
        "project": {"name": "test-project"},
        "data": {
            "target": "churn",
            "id_column": "customer_id",
        },
        "training": {"cv_folds": 2, "validation_size": 0.2, "test_size": 0.2},
        "model": {"algorithm": "random_forest", "params": {"n_estimators": 50}},
    }
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "config.yaml").write_text(
        yaml.safe_dump(cfg), encoding="utf-8"
    )
    for d in ("data/raw", "data/interim", "data/processed", "models"):
        (root / d).mkdir(parents=True, exist_ok=True)


def test_init_creates_project(tmp_path):
    target = tmp_path / "my-project"
    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 0, result.output
    assert (target / "configs" / "config.yaml").exists()
    assert (target / "data" / "raw").is_dir()
    assert (target / "models").is_dir()
    assert (target / ".gitignore").exists()


def test_init_with_demo(tmp_path):
    target = tmp_path / "demo-project"
    result = runner.invoke(app, ["init", str(target), "--demo"])
    assert result.exit_code == 0, result.output
    demo = target / "data" / "raw" / "train.csv"
    assert demo.exists()
    df = pd.read_csv(demo)
    assert "churn" in df.columns
    cfg = yaml.safe_load((target / "configs" / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["data"]["target"] == "churn"


def test_init_refuses_to_overwrite(tmp_path):
    target = tmp_path / "p"
    runner.invoke(app, ["init", str(target)])
    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 1
    assert "already exists" in result.output


def test_status_requires_project(tmp_path):
    result = runner.invoke(app, ["status", "--project-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "Not inside a dscli project" in result.output


def test_data_info_standalone(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")
    result = runner.invoke(app, ["data", "info", str(path)])
    assert result.exit_code == 0, result.output
    assert "2 rows" in result.output.replace("\n", " ").replace("  ", " ")


def test_data_info_missing_file(tmp_path):
    result = runner.invoke(app, ["data", "info", str(tmp_path / "nope.csv")])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_train_end_to_end(tmp_path):
    """The full workflow: init -> clean -> split -> train -> evaluate -> predict."""
    project = tmp_path / "project"
    runner.invoke(app, ["init", str(project), "--demo"])
    _write_project_config(project)

    # clean
    result = runner.invoke(
        app,
        [
            "data", "clean",
            "--project-dir", str(project),
            "--input", "data/raw/train.csv",
            "--output", "data/interim/train.csv",
            "--overwrite",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (project / "data" / "interim" / "train.csv").exists()

    # split
    result = runner.invoke(app, ["split", "--project-dir", str(project)])
    assert result.exit_code == 0, result.output
    assert (project / "data" / "processed" / "train.csv").exists()
    assert (project / "data" / "processed" / "validation.csv").exists()
    assert (project / "data" / "processed" / "test.csv").exists()

    # train
    result = runner.invoke(
        app, ["train", "--project-dir", str(project), "--overwrite"]
    )
    assert result.exit_code == 0, result.output
    model_path = project / "models" / "random_forest.joblib"
    assert model_path.exists()
    assert "Accuracy" in result.output

    # evaluate
    result = runner.invoke(app, ["evaluate", "--project-dir", str(project)])
    assert result.exit_code == 0, result.output
    eval_path = project / "reports" / "reports" / "evaluation_random_forest.json"
    assert eval_path.exists()
    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    # Labels must survive the save/load round-trip; accuracy on a held-out
    # test set should be far above chance (0.5), not 0.
    assert eval_data["metrics"]["accuracy"] > 0.6, eval_data["metrics"]

    # predict
    result = runner.invoke(
        app,
        [
            "predict",
            "--project-dir", str(project),
            "--input", "data/processed/test.csv",
            "--output", "predictions.csv",
            "--overwrite",
        ],
    )
    assert result.exit_code == 0, result.output
    preds = pd.read_csv(project / "predictions.csv")
    assert "prediction" in preds.columns
    assert len(preds) == len(pd.read_csv(project / "data" / "processed" / "test.csv"))

    # experiments were recorded
    result = runner.invoke(app, ["experiments", "list", "--project-dir", str(project)])
    assert result.exit_code == 0, result.output
    assert "random_forest" in result.output

    # status
    result = runner.invoke(app, ["status", "--project-dir", str(project)])
    assert result.exit_code == 0, result.output
    assert "test-project" in result.output

    # report
    result = runner.invoke(app, ["report", "--project-dir", str(project), "--overwrite"])
    assert result.exit_code == 0, result.output
    report = project / "reports" / "reports" / "report.md"
    assert report.exists()
    assert "test-project" in report.read_text(encoding="utf-8")


def test_compare_records_experiments(tmp_path):
    project = tmp_path / "project"
    runner.invoke(app, ["init", str(project), "--demo"])
    _write_project_config(project)
    runner.invoke(
        app,
        [
            "data", "clean",
            "--project-dir", str(project),
            "--input", "data/raw/train.csv",
            "--output", "data/interim/train.csv",
            "--overwrite",
        ],
    )
    runner.invoke(app, ["split", "--project-dir", str(project)])

    result = runner.invoke(
        app,
        [
            "compare",
            "--project-dir", str(project),
            "--models", "logistic_regression,random_forest",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Model Comparison" in result.output
    assert (project / "reports" / "reports" / "comparison.json").exists()


def test_train_unknown_model_fails(tmp_path):
    project = tmp_path / "project"
    runner.invoke(app, ["init", str(project), "--demo"])
    _write_project_config(project)
    result = runner.invoke(
        app,
        ["train", "--project-dir", str(project), "--model", "knn_magic"],
    )
    assert result.exit_code == 1
    assert "Unknown model" in result.output


def test_experiments_show_and_delete(tmp_path):
    project = tmp_path / "project"
    runner.invoke(app, ["init", str(project), "--demo"])
    _write_project_config(project)
    runner.invoke(
        app,
        [
            "data", "clean",
            "--project-dir", str(project),
            "--input", "data/raw/train.csv",
            "--output", "data/interim/train.csv",
            "--overwrite",
        ],
    )
    runner.invoke(app, ["split", "--project-dir", str(project)])
    runner.invoke(app, ["train", "--project-dir", str(project)])

    import re

    result = runner.invoke(app, ["experiments", "list", "--project-dir", str(project)])
    match = re.search(r"\b([0-9a-f]{12})\b", result.output)
    assert match, result.output
    exp_id = match.group(1)
    result = runner.invoke(app, ["experiments", "show", exp_id, "--project-dir", str(project)])
    assert result.exit_code == 0, result.output
    assert "random_forest" in result.output

    result = runner.invoke(app, ["experiments", "delete", exp_id, "--project-dir", str(project)])
    assert result.exit_code == 0, result.output
