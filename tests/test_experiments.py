"""Tests for the experiment tracker."""

from __future__ import annotations

from pathlib import Path

import pytest

from dscli.errors import ExperimentError
from dscli.experiments.tracker import ExperimentTracker


@pytest.fixture
def tracker(tmp_path: Path) -> ExperimentTracker:
    return ExperimentTracker(tmp_path / "experiments.db")


def test_record_and_get(tracker):
    exp_id = tracker.record(
        model="random_forest",
        task="classification",
        dataset="data/train.csv",
        hyperparameters={"n_estimators": 100},
        metrics={"accuracy": 0.9},
        training_duration=1.5,
    )
    exp = tracker.get(exp_id)
    assert exp.id == exp_id
    assert exp.model == "random_forest"
    assert exp.metrics["accuracy"] == 0.9
    assert exp.hyperparameters["n_estimators"] == 100
    assert exp.primary_metric == ("accuracy", 0.9)


def test_list_orders_newest_first(tracker):
    id1 = tracker.record(model="a", task="classification")
    id2 = tracker.record(model="b", task="regression")
    experiments = tracker.list()
    assert [e.id for e in experiments][:2] == [id2, id1]


def test_get_missing_raises(tracker):
    with pytest.raises(ExperimentError, match="not found"):
        tracker.get("does-not-exist")


def test_delete(tracker):
    exp_id = tracker.record(model="a", task="classification")
    tracker.delete(exp_id)
    assert tracker.count() == 0
    with pytest.raises(ExperimentError):
        tracker.delete(exp_id)


def test_export(tracker, tmp_path):
    tracker.record(model="ridge", task="regression", metrics={"r2": 0.8})
    out = tracker.export(tmp_path / "export.json")
    assert out.exists()
    import json

    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["model"] == "ridge"


def test_persistence_across_instances(tmp_path):
    db = tmp_path / "experiments.db"
    with ExperimentTracker(db) as t1:
        exp_id = t1.record(model="xgb", task="classification")
    with ExperimentTracker(db) as t2:
        assert t2.get(exp_id).model == "xgb"
