"""Experiment tracking backed by SQLite.

Every training run is recorded as an experiment with its model, task,
hyperparameters, dataset, metrics, duration, and model path. The database
lives at ``experiments.db`` in the project root (configurable via
``experiments.db``) and requires no external services.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dscli.errors import ExperimentError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    model TEXT NOT NULL,
    task TEXT NOT NULL,
    dataset TEXT,
    hyperparameters TEXT,
    metrics TEXT,
    cv_scores TEXT,
    training_duration REAL,
    model_path TEXT,
    target TEXT,
    n_train INTEGER,
    n_validation INTEGER,
    note TEXT
);
"""


@dataclass
class Experiment:
    """A single recorded training run."""

    id: str
    timestamp: str
    model: str
    task: str
    dataset: str | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    cv_scores: dict[str, Any] = field(default_factory=dict)
    training_duration: float | None = None
    model_path: str | None = None
    target: str | None = None
    n_train: int | None = None
    n_validation: int | None = None
    note: str | None = None

    @property
    def primary_metric(self) -> tuple[str, float] | None:
        """The headline metric for the task, if available."""
        key = "accuracy" if self.task == "classification" else "r2"
        if key in self.metrics:
            return key, self.metrics[key]
        return None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Experiment":
        def _json(value: str | None, default: Any) -> Any:
            if not value:
                return default
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return default

        return cls(
            id=row["id"],
            timestamp=row["timestamp"],
            model=row["model"],
            task=row["task"],
            dataset=row["dataset"],
            hyperparameters=_json(row["hyperparameters"], {}),
            metrics=_json(row["metrics"], {}),
            cv_scores=_json(row["cv_scores"], {}),
            training_duration=row["training_duration"],
            model_path=row["model_path"],
            target=row["target"],
            n_train=row["n_train"],
            n_validation=row["n_validation"],
            note=row["note"],
        )


class ExperimentTracker:
    """SQLite-backed experiment store."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute(_SCHEMA)
            self._conn.commit()
        except sqlite3.Error as exc:
            raise ExperimentError(
                f"Could not open experiment database '{self.db_path}': {exc}"
            ) from exc

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- write -------------------------------------------------------------

    def record(
        self,
        *,
        model: str,
        task: str,
        dataset: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        cv_scores: dict[str, Any] | None = None,
        training_duration: float | None = None,
        model_path: str | None = None,
        target: str | None = None,
        n_train: int | None = None,
        n_validation: int | None = None,
        note: str | None = None,
    ) -> str:
        """Insert an experiment and return its generated id."""
        exp_id = uuid.uuid4().hex[:12]
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            self._conn.execute(
                """
                INSERT INTO experiments (
                    id, timestamp, model, task, dataset, hyperparameters, metrics,
                    cv_scores, training_duration, model_path, target, n_train,
                    n_validation, note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exp_id,
                    timestamp,
                    model,
                    task,
                    dataset,
                    json.dumps(hyperparameters or {}, default=str),
                    json.dumps(metrics or {}, default=str),
                    json.dumps(cv_scores or {}, default=str),
                    training_duration,
                    model_path,
                    target,
                    n_train,
                    n_validation,
                    note,
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise ExperimentError(f"Failed to record experiment: {exc}") from exc
        return exp_id

    # -- read --------------------------------------------------------------

    def list(self, limit: int = 50) -> list[Experiment]:
        """Most recent experiments first."""
        try:
            rows = self._conn.execute(
                "SELECT * FROM experiments ORDER BY timestamp DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise ExperimentError(f"Failed to list experiments: {exc}") from exc
        return [Experiment.from_row(row) for row in rows]

    def get(self, experiment_id: str) -> Experiment:
        """Fetch a single experiment by id."""
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            raise ExperimentError(
                f"Experiment '{experiment_id}' not found. "
                "Run 'dscli experiments list' to see available ids."
            )
        return Experiment.from_row(row)

    def delete(self, experiment_id: str) -> None:
        """Delete an experiment by id."""
        cursor = self._conn.execute(
            "DELETE FROM experiments WHERE id = ?", (experiment_id,)
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            raise ExperimentError(f"Experiment '{experiment_id}' not found.")

    def count(self) -> int:
        """Number of recorded experiments."""
        row = self._conn.execute("SELECT COUNT(*) AS n FROM experiments").fetchone()
        return int(row["n"])

    def export(self, path: str | Path) -> Path:
        """Export all experiments to a JSON file; returns the written path."""
        experiments = [asdict(exp) for exp in self.list(limit=10_000)]
        from dscli.utils.io import save_json

        return save_json(experiments, path)
