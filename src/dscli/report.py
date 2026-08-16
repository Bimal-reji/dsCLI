"""Markdown report generation.

``generate_report`` combines a dataset summary, validation findings, model
metrics, feature importances, and figure references into a single Markdown
document that a data scientist can drop into a notebook or share.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from dscli.config import Config
from dscli.data.loader import describe_dataset
from dscli.data.validator import validate_dataframe, validate_target
from dscli.evaluation.metrics import METRIC_LABELS
from dscli.models.persistence import load_model
from dscli.utils.io import check_can_write


def _metric_rows(metrics: dict) -> list[tuple[str, str]]:
    rows = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            label = METRIC_LABELS.get(key, key.replace("_", " ").title())
            rows.append((label, f"{value:.4f}"))
    return rows


def generate_report(
    config: Config,
    *,
    data_path: Path,
    model_path: Path | None = None,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    """Write a Markdown report to ``output_path``; returns the written path."""
    check_can_write(output_path, overwrite, description="report")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# Project Report — {config.project.name}")
    lines.append("")
    lines.append(
        f"_Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_"
    )
    lines.append("")

    # -- dataset section ----------------------------------------------------
    lines.append("## Dataset")
    lines.append("")
    df = pd.read_csv(data_path) if data_path.exists() else None
    if df is not None:
        info = describe_dataset(df)
        lines.append(f"- **Source**: `{data_path}`")
        lines.append(f"- **Rows**: {info['rows']}")
        lines.append(f"- **Columns**: {info['columns']}")
        lines.append(f"- **Missing values**: {info['missing_values']} ({info['missing_pct']}%)")
        lines.append(f"- **Duplicate rows**: {info['duplicate_rows']}")
        lines.append("")

        target = config.data.target
        report = validate_dataframe(df, required_columns=[target] if target else None)
        if report.errors:
            lines.append("### Data quality problems")
            lines.append("")
            for error in report.errors:
                lines.append(f"- ⚠️ {error}")
            lines.append("")

    # -- model section -------------------------------------------------------
    if model_path is not None and model_path.exists():
        try:
            _, metadata = load_model(model_path)
        except Exception:
            metadata = None
        if metadata is not None:
            lines.append("## Model")
            lines.append("")
            lines.append(f"- **Algorithm**: `{metadata.model_name}`")
            lines.append(f"- **Task**: {metadata.task}")
            lines.append(f"- **Target**: `{metadata.target}`")
            lines.append(f"- **Artifact**: `{model_path}`")
            lines.append("")
            metric_rows = _metric_rows(metadata.metrics)
            if metric_rows:
                lines.append("| Metric | Score |")
                lines.append("| --- | --- |")
                for label, score in metric_rows:
                    lines.append(f"| {label} | {score} |")
                lines.append("")
            if metadata.cv_scores:
                cv = metadata.cv_scores
                lines.append(
                    f"- **Cross-validation ({'cv'}): mean = {cv.get('cv_mean', 'n/a')} "
                    f"± {cv.get('cv_std', 'n/a')}"
                )
                lines.append("")
            if metadata.feature_importance:
                lines.append("### Top features")
                lines.append("")
                lines.append("| Feature | Importance |")
                lines.append("| --- | --- |")
                for name, score in metadata.feature_importance[:15]:
                    lines.append(f"| {name} | {score} |")
                lines.append("")

    # -- figures section -----------------------------------------------------
    figure_dir = config.figure_dir
    figures = sorted(figure_dir.glob("*.png")) if figure_dir.exists() else []
    if figures:
        lines.append("## Figures")
        lines.append("")
        for figure in figures:
            rel = figure.relative_to(config.project_root)
            lines.append(f"![{figure.stem}]({rel.as_posix()})")
        lines.append("")

    lines.append("---")
    lines.append("_Generated with dscli._")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
