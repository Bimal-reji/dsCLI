"""dscli command-line interface.

The CLI layer is deliberately thin: it parses arguments, validates them,
calls into the business-logic modules, and renders results with Rich.
All meaningful errors are converted into clean, user-facing messages with a
non-zero exit code.
"""

from __future__ import annotations

import functools
import sys
import traceback
from pathlib import Path
from typing import Annotated, Any, Optional

import numpy as np
import pandas as pd
import typer
from rich import box
from rich.panel import Panel
from rich.table import Table

from dscli.config import (
    Config,
    default_config_path,
    ensure_project_init,
    find_project_root,
    load_project_config,
)
from dscli.data.cleaner import clean_dataframe
from dscli.data.demo import generate_demo_dataset
from dscli.data.loader import (
    describe_dataset,
    load_dataframe,
    save_dataframe,
)
from dscli.data.validator import (
    check_class_balance,
    validate_dataframe,
    validate_target,
)
from dscli.errors import DScliError, DataError, ModelError
from dscli.evaluation.metrics import (
    METRIC_LABELS,
    compute_metrics,
    extract_feature_importance,
)
from dscli.experiments.tracker import ExperimentTracker
from dscli.features.builder import FeatureSpec, build_preprocessor
from dscli.models import available_models, load_model, save_model, train_model
from dscli.models.trainer import TrainingResult, predict_with_pipeline
from dscli.report import generate_report
from dscli.utils.console import (
    console,
    error_panel,
    metrics_table,
    print_info,
    print_success,
    print_warning,
    summary_table,
)
from dscli.utils.io import check_can_write, save_json
from dscli.utils.logging_utils import get_logger, setup_logging
from dscli.visualization.plots import generate_figures

def _force_utf8_stdio() -> None:
    """Reconfigure stdout/stderr to UTF-8 so unicode glyphs (✓, ─, …) never
    crash on consoles with legacy encodings (e.g. cp1252 on Windows)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


_force_utf8_stdio()

app = typer.Typer(
    help="Data Science CLI — manage the complete ML workflow from the terminal.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

data_app = typer.Typer(help="Inspect, validate, and clean datasets.", no_args_is_help=True)
app.add_typer(data_app, name="data")

features_app = typer.Typer(help="Build engineered features.", no_args_is_help=True)
app.add_typer(features_app, name="features")

experiments_app = typer.Typer(help="Track and inspect training runs.", no_args_is_help=True)
app.add_typer(experiments_app, name="experiments")

config_app = typer.Typer(help="Inspect project configuration.", no_args_is_help=True)
app.add_typer(config_app, name="config")

# Shared definitions for the global options attached to every command.
PROJECT_DIR_OPT = typer.Option(
    "--project-dir",
    help="Path to the dscli project root (default: nearest project found upward).",
)
CONFIG_OPT = typer.Option("--config", help="Path to a YAML configuration file.")
VERBOSE_OPT = typer.Option(
    "--verbose", "-v", help="Enable debug logging and tracebacks."
)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def _handle_error(exc: BaseException, verbose: bool) -> None:
    """Render an error to the console and raise typer.Exit(1)."""
    if isinstance(exc, DScliError):
        error_panel(str(exc))
        get_logger().error("Command failed: %s", exc)
    else:
        get_logger().error("Unexpected error: %s: %s", type(exc).__name__, exc, exc_info=True)
        message = f"[red]Unexpected error: {type(exc).__name__}: {exc}[/red]"
        if verbose:
            message += f"\n\n[dim]{traceback.format_exc()}[/dim]"
        else:
            message += "\n\nSee logs/dscli.log for details, or rerun with --verbose."
        error_panel(message)
    raise typer.Exit(code=1) from exc


def command(func):
    """Decorator: set up logging + context, and convert errors into clean
    messages with exit code 1."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Typer passes the click context through the ``ctx`` keyword. In Typer
        # >= 0.26 the runtime object is the vendored click Context, so
        # ``isinstance(ctx, typer.Context)`` would be False; duck-type instead.
        ctx = kwargs.get("ctx") or (args[0] if args else None)
        project_dir = kwargs.get("project_dir")
        config = kwargs.get("config")
        verbose = bool(kwargs.get("verbose", False))

        if ctx is not None and hasattr(ctx, "obj"):
            root = Path(project_dir).resolve() if project_dir else find_project_root()
            log_dir = root / "logs" if root else Path.cwd() / "logs"
            setup_logging(log_dir, verbose=verbose)
            ctx.obj = {
                "project_dir": project_dir,
                "config": config,
                "verbose": verbose,
            }

        try:
            return func(*args, **kwargs)
        except typer.Exit:
            raise
        except (DScliError, Exception) as exc:
            _handle_error(exc, verbose)

    return wrapper


def _ctx_config(ctx: typer.Context) -> tuple[Config, Path]:
    """Load the effective config from the context's global options."""
    opts = ctx.obj or {}
    config, root = load_project_config(
        config_path=opts.get("config"), project_dir=opts.get("project_dir")
    )
    return config, root


def _default_input(config: Config, kind: str) -> Path:
    """Resolve a conventional default dataset path."""
    return config.resolve(kind)


def _resolve_model_path(config: Config, model_path: str | Path | None) -> Path:
    """Find the model artifact to use for evaluate/predict/report."""
    if model_path is not None:
        return config.resolve(str(model_path))
    default = config.model_dir / f"{config.model.algorithm}.joblib"
    if default.exists():
        return default
    candidates = sorted(config.model_dir.glob("*.joblib"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ModelError(
            f"Multiple models found in '{config.model_dir}'. Pass --model to choose one."
        )
    raise ModelError(
        f"No trained model found in '{config.model_dir}'. Run 'dscli train' first, "
        "or pass --model with a path to a saved .joblib artifact."
    )


def _load_training_data(config: Config, df_path: Path) -> pd.DataFrame:
    df = load_dataframe(df_path)
    # Drop the id column up front; it carries no signal.
    if config.data.id_column and config.data.id_column in df.columns:
        df = df.drop(columns=[config.data.id_column])
    return df


# ---------------------------------------------------------------------------
# Project scaffolding
# ---------------------------------------------------------------------------

_PROJECT_GITIGNORE = """\
# Python
__pycache__/
*.py[cod]
.venv/

# Data
data/raw/*
data/interim/*
data/processed/*
data/external/*
!data/**/.gitkeep

# Artifacts
models/*
!models/.gitkeep
reports/figures/*
!reports/figures/.gitkeep
logs/*
!logs/.gitkeep
experiments.db
"""


@app.command(help="Scaffold a new dscli project.")
@command
def init(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Project name. Creates a subdirectory of that name; defaults to the current directory."),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing configuration.")
    ] = False,
    demo: Annotated[
        bool,
        typer.Option("--demo", help="Also generate a demo dataset and configure the target column."),
    ] = False,
    project_dir: Annotated[Optional[Path], PROJECT_DIR_OPT] = None,
    config: Annotated[Optional[Path], CONFIG_OPT] = None,
    verbose: Annotated[bool, VERBOSE_OPT] = False,
) -> None:
    root = Path(name).resolve() if name else Path.cwd().resolve()
    config_path = root / "configs" / "config.yaml"
    if config_path.exists() and not force:
        raise DataError(
            f"A project already exists at '{root}' (found '{config_path}'). "
            "Use --force to re-initialize."
        )

    config = Config.from_dict({}, project_root=root)
    ensure_project_init(config)
    (root / "configs").mkdir(parents=True, exist_ok=True)

    # Preserve placeholder files so the empty dirs survive git.
    for directory in (
        config.data_dir("raw"),
        config.data_dir("interim"),
        config.data_dir("processed"),
        config.data_dir("external"),
        config.model_dir,
        config.figure_dir,
        config.log_dir,
        config.resolve("notebooks"),
    ):
        (directory / ".gitkeep").write_text("", encoding="utf-8")

    config.dump(config_path)
    (root / ".gitignore").write_text(_PROJECT_GITIGNORE, encoding="utf-8")

    if demo:
        demo_path = config.data_dir("raw") / "train.csv"
        df = generate_demo_dataset()
        save_dataframe(df, demo_path, overwrite=True)
        config = config.with_overrides({"data": {"target": "churn", "id_column": "customer_id"}})
        config.dump(config_path)
        print_success(f"Demo dataset written to '{demo_path}' with target 'churn'.")

    console.print(
        Panel(
            f"[green]Project initialized at[/green] [bold]{root}[/bold]\n\n"
            f"Configuration:  {config_path}\n"
            f"Next steps:\n"
            f"  dscli data info data/raw/train.csv\n"
            f"  dscli data validate\n"
            f"  dscli data clean\n"
            f"  dscli split\n"
            f"  dscli train\n"
            f"  dscli evaluate\n"
            f"  dscli report",
            title="[bold]✓ Project ready[/bold]",
            border_style="green",
            expand=False,
        )
    )


@app.command(help="Show project status and effective configuration.")
@command
def status(
    ctx: typer.Context,
    project_dir: Annotated[Optional[Path], PROJECT_DIR_OPT] = None,
    config: Annotated[Optional[Path], CONFIG_OPT] = None,
    verbose: Annotated[bool, VERBOSE_OPT] = False,
) -> None:
    config, root = _ctx_config(ctx)

    datasets = {
        "raw train": config.data_dir("raw") / "train.csv",
        "interim train": config.data_dir("interim") / "train.csv",
        "processed train": config.resolve(config.data.train),
        "processed validation": config.resolve(config.data.validation),
        "processed test": config.resolve(config.data.test),
    }
    dataset_rows = [(label, "✓" if path.exists() else "—") for label, path in datasets.items()]

    model_files = sorted(config.model_dir.glob("*.joblib"))
    with ExperimentTracker(config.experiments_db) as tracker:
        experiment_count = tracker.count()

    rows = [
        ("Project name", config.project.name),
        ("Project root", str(root)),
        ("Config file", str(default_config_path(root))),
        ("Target column", config.data.target or "not set"),
        ("Model algorithm", config.model.algorithm),
        ("Task type", "auto (inferred from target)"),
        ("Models saved", f"{len(model_files)} artifact(s)"),
        ("Experiments", str(experiment_count)),
    ]
    rows += dataset_rows
    console.print(summary_table("Project Status", rows))
    console.print()
    print_info(f"Run 'dscli config show' for the full configuration.")


@config_app.command("show", help="Print the effective configuration as YAML.")
@command
def config_show(
    ctx: typer.Context,
    project_dir: Annotated[Optional[Path], PROJECT_DIR_OPT] = None,
    config: Annotated[Optional[Path], CONFIG_OPT] = None,
    verbose: Annotated[bool, VERBOSE_OPT] = False,
) -> None:
    import yaml

    config, _ = _ctx_config(ctx)
    console.print(yaml.safe_dump(config.to_dict(), sort_keys=False))


# ---------------------------------------------------------------------------
# Data commands
# ---------------------------------------------------------------------------


@data_app.command("info", help="Show a summary of a dataset.")
@command
def data_info(
    path: Annotated[Path, typer.Argument(help="Path to the dataset.")],
    project_dir: Annotated[Optional[Path], PROJECT_DIR_OPT] = None,
    config: Annotated[Optional[Path], CONFIG_OPT] = None,
    verbose: Annotated[bool, VERBOSE_OPT] = False,
) -> None:
    df = load_dataframe(path)
    info = describe_dataset(df)

    console.print(
        Panel(
            f"[bold]{path}[/bold]\n"
            f"[cyan]{info['rows']:,}[/cyan] rows × [cyan]{info['columns']}[/cyan] columns",
            title="[bold]Dataset loaded[/bold]",
            border_style="green",
            expand=False,
        )
    )
    console.print()
    rows = [
        ("Rows", f"{info['rows']:,}"),
        ("Columns", str(info["columns"])),
        ("Numeric features", str(len(info["numeric_columns"]))),
        ("Categorical features", str(len(info["categorical_columns"]))),
        ("Missing values", f"{info['missing_values']:,} ({info['missing_pct']}%)"),
        ("Duplicate rows", f"{info['duplicate_rows']:,}"),
        ("Memory", f"{info['memory_mb']} MB"),
    ]
    console.print(summary_table("Dataset Summary", rows))
    console.print()
    table = Table(title="Columns", box=box.SIMPLE_HEAD, show_header=True)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Column", style="cyan")
    table.add_column("Dtype", style="magenta")
    table.add_column("Non-null", justify="right")
    table.add_column("Unique", justify="right")
    for idx, col in enumerate(df.columns, start=1):
        table.add_row(
            str(idx),
            col,
            str(df[col].dtype),
            f"{df[col].notna().sum():,}",
            f"{df[col].nunique():,}",
        )
    console.print(table)


@data_app.command("validate", help="Validate a dataset against project expectations.")
@command
def data_validate(
    ctx: typer.Context,
    input: Annotated[
        Optional[Path],
        typer.Option("--input", "-i", help="Dataset to validate (default: data/raw/train.csv)."),
    ] = None,
    target: Annotated[
        Optional[str], typer.Option("--target", help="Override the configured target column.")
    ] = None,
    project_dir: Annotated[Optional[Path], PROJECT_DIR_OPT] = None,
    config: Annotated[Optional[Path], CONFIG_OPT] = None,
    verbose: Annotated[bool, VERBOSE_OPT] = False,
) -> None:
    config, _ = _ctx_config(ctx)
    path = config.resolve(str(input)) if input else _default_input(config, "data/raw/train.csv")
    df = load_dataframe(path)

    target_name = target or config.data.target
    report = validate_dataframe(df, required_columns=[target_name] if target_name else None)

    console.print(
        Panel(
            f"[bold]{path}[/bold] — {report.rows:,} rows × {report.columns} columns",
            title="[bold]Validation report[/bold]",
            border_style="green" if report.is_valid else "yellow",
            expand=False,
        )
    )
    for error in report.errors:
        print_error(error)
    for warning in report.warnings:
        print_warning(warning)

    if target_name:
        try:
            _, task = validate_target(df, target_name)
            print_success(f"Target '{target_name}' present (task: {task}).")
            if task == "classification":
                balance = check_class_balance(df, target_name)
                console.print()
                balance_table = Table(title="Class balance", box=box.SIMPLE_HEAD)
                balance_table.add_column("Class", style="cyan")
                balance_table.add_column("Count", justify="right")
                for cls, count in balance.items():
                    balance_table.add_row(str(cls), f"{count:,}")
                console.print(balance_table)
        except DScliError as exc:
            print_error(str(exc))

    if not report.is_valid:
        raise typer.Exit(code=1)
    if report.is_valid and not report.errors:
        print_success("Dataset is valid.")


@data_app.command("clean", help="Clean a dataset and write it to a new location.")
@command
def data_clean(
    ctx: typer.Context,
    input: Annotated[
        Optional[Path],
        typer.Option("--input", "-i", help="Dataset to clean (default: data/raw/train.csv)."),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Where to write the cleaned dataset (default: data/interim/train.csv)."),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite an existing output file.")
    ] = False,
    project_dir: Annotated[Optional[Path], PROJECT_DIR_OPT] = None,
    config: Annotated[Optional[Path], CONFIG_OPT] = None,
    verbose: Annotated[bool, VERBOSE_OPT] = False,
) -> None:
    config, _ = _ctx_config(ctx)
    in_path = config.resolve(str(input)) if input else _default_input(config, "data/raw/train.csv")
    out_path = config.resolve(str(output)) if output else _default_input(config, "data/interim/train.csv")

    df = load_dataframe(in_path)
    check_can_write(out_path, overwrite or config.output.overwrite, description="output dataset")

    cleaned, report = clean_dataframe(
        df,
        drop_duplicates=config.cleaning.drop_duplicates,
        missing_strategy=config.cleaning.missing_strategy,
        missing_constant=config.cleaning.missing_constant,
        drop_high_missing_threshold=config.cleaning.drop_high_missing_threshold,
        outlier_method=config.cleaning.outlier_method,
        outlier_threshold=config.cleaning.outlier_threshold,
    )
    save_dataframe(cleaned, out_path, overwrite=True)

    rows = [
        ("Rows before", f"{report.rows_before:,}"),
        ("Rows after", f"{report.rows_after:,}"),
        ("Duplicates dropped", f"{report.duplicates_dropped:,}"),
        ("Rows dropped (missing)", f"{report.rows_dropped_missing:,}"),
        ("Missing values imputed", f"{report.missing_imputed:,}"),
        ("Outliers capped", f"{report.outliers_capped:,}"),
    ]
    if report.columns_dropped:
        rows.append(("Columns dropped", ", ".join(report.columns_dropped)))
    console.print(summary_table("Cleaning Summary", rows))
    for note in report.notes or []:
        print_info(note)
    print_success(f"Cleaned dataset written to '{out_path}'.")


# ---------------------------------------------------------------------------
# EDA / features / split
# ---------------------------------------------------------------------------


@app.command(help="Exploratory data analysis: summary + figures.")
@command
def eda(
    ctx: typer.Context,
    input: Annotated[
        Optional[Path],
        typer.Option("--input", "-i", help="Dataset to analyze (default: data/interim/train.csv)."),
    ] = None,
    target: Annotated[
        Optional[str], typer.Option("--target", help="Override the configured target column.")
    ] = None,
    project_dir: Annotated[Optional[Path], PROJECT_DIR_OPT] = None,
    config: Annotated[Optional[Path], CONFIG_OPT] = None,
    verbose: Annotated[bool, VERBOSE_OPT] = False,
) -> None:
    config, _ = _ctx_config(ctx)
    in_path = config.resolve(str(input)) if input else _default_input(config, "data/interim/train.csv")
    df = load_dataframe(in_path)
    target_name = target or config.data.target

    print_info(f"Running EDA on '{in_path}'...")
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        console.print(numeric.describe().T.round(4))
        console.print()

    report = validate_dataframe(df, required_columns=[target_name] if target_name else None)
    for warning in report.warnings:
        print_warning(warning)

    figures = generate_figures(
        df, config.figure_dir, target=target_name if target_name in df.columns else None
    )
    console.print()
    print_success(f"Generated {len(figures)} figure(s) in '{config.figure_dir}'.")
    for fig in figures:
        print_info(f"  {fig}")


@features_app.command("build", help="Build a feature matrix and reusable preprocessor.")
@command
def features_build(
    ctx: typer.Context,
    input: Annotated[
        Optional[Path],
        typer.Option("--input", "-i", help="Dataset to transform (default: data/interim/train.csv)."),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Feature matrix output (default: data/processed/features.csv)."),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite existing outputs.")
    ] = False,
    project_dir: Annotated[Optional[Path], PROJECT_DIR_OPT] = None,
    config: Annotated[Optional[Path], CONFIG_OPT] = None,
    verbose: Annotated[bool, VERBOSE_OPT] = False,
) -> None:
    config, _ = _ctx_config(ctx)
    in_path = config.resolve(str(input)) if input else _default_input(config, "data/interim/train.csv")
    out_path = config.resolve(str(output)) if output else _default_input(config, "data/processed/features.csv")

    df = load_dataframe(in_path)
    target_name = config.data.target
    exclude = [target_name] if target_name and target_name in df.columns else []
    if config.data.id_column:
        exclude.append(config.data.id_column)

    spec = FeatureSpec.from_dataframe(
        df.drop(columns=[c for c in exclude if c in df.columns]),
        max_categories=config.features.max_categories,
    )
    preprocessor = build_preprocessor(
        spec,
        scale_numerical=config.features.scale_numerical,
        scaler=config.features.scaler,
        categorical_encoding=config.features.categorical_encoding,
        handle_unknown=config.features.handle_unknown,
    )

    X = df[spec.feature_columns]
    transformed = preprocessor.fit_transform(X)
    feature_names = list(preprocessor.get_feature_names_out())
    feature_df = pd.DataFrame(transformed, columns=feature_names, index=df.index)
    if target_name and target_name in df.columns:
        feature_df[target_name] = df[target_name]

    check_can_write(out_path, overwrite or config.output.overwrite, description="feature matrix")
    save_dataframe(feature_df, out_path, overwrite=True)
    from dscli.utils.io import save_model_artifact

    preprocessor_path = config.model_dir / "preprocessor.joblib"
    check_can_write(
        preprocessor_path, overwrite or config.output.overwrite, description="preprocessor"
    )
    save_model_artifact(preprocessor, preprocessor_path, overwrite=True)

    console.print(
        summary_table(
            "Feature Spec",
            [
                ("Numeric features", str(len(spec.numeric_columns))),
                ("Categorical features", str(len(spec.categorical_columns))),
                ("Dropped columns", ", ".join(spec.dropped_columns) or "none"),
                ("Transformed features", str(len(feature_names))),
            ],
        )
    )
    print_success(f"Feature matrix written to '{out_path}'.")
    print_success(f"Preprocessor saved to '{preprocessor_path}'.")


@app.command(help="Split cleaned data into train/validation/test sets.")
@command
def split(
    ctx: typer.Context,
    input: Annotated[
        Optional[Path],
        typer.Option("--input", "-i", help="Dataset to split (default: data/interim/train.csv)."),
    ] = None,
    target: Annotated[
        Optional[str], typer.Option("--target", help="Override the configured target column.")
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite existing split files.")
    ] = False,
    project_dir: Annotated[Optional[Path], PROJECT_DIR_OPT] = None,
    config: Annotated[Optional[Path], CONFIG_OPT] = None,
    verbose: Annotated[bool, VERBOSE_OPT] = False,
) -> None:
    from sklearn.model_selection import train_test_split

    config, _ = _ctx_config(ctx)
    in_path = config.resolve(str(input)) if input else _default_input(config, "data/interim/train.csv")
    df = load_dataframe(in_path)

    if config.training.test_size >= 1.0 or config.training.validation_size >= 1.0:
        raise DataError(
            "Invalid training split: test_size and validation_size must each be less than 1.0."
        )
    if config.training.test_size + config.training.validation_size >= 1.0:
        raise DataError(
            "Invalid training split: test_size + validation_size must be less than 1.0."
        )

    target_name = target or config.data.target
    validate_target(df, target_name)
    _, task = validate_target(df, target_name)

    stratify = df[target_name] if config.training.stratify and task == "classification" else None
    if stratify is not None and (stratify.nunique() < 2 or stratify.value_counts().min() < 2):
        stratify = None

    rest, test = train_test_split(
        df,
        test_size=config.training.test_size,
        random_state=config.training.random_state,
        stratify=stratify,
    )
    val_fraction = config.training.validation_size / (1.0 - config.training.test_size)
    train, validation = train_test_split(
        rest,
        test_size=val_fraction,
        random_state=config.training.random_state,
        stratify=rest[target_name] if stratify is not None else None,
    )

    for split_df, split_path in (
        (train, config.resolve(config.data.train)),
        (validation, config.resolve(config.data.validation)),
        (test, config.resolve(config.data.test)),
    ):
        check_can_write(split_path, overwrite or config.output.overwrite, description="split file")
        save_dataframe(split_df, split_path, overwrite=True)

    console.print(
        summary_table(
            "Split Summary",
            [
                ("Train", f"{len(train):,} rows"),
                ("Validation", f"{len(validation):,} rows"),
                ("Test", f"{len(test):,} rows"),
                ("Total", f"{len(df):,} rows"),
            ],
        )
    )
    print_success(
        f"Wrote train/validation/test to '{config.resolve(config.data.train).parent}'."
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _print_training_result(result: TrainingResult, model_path: Path | None = None) -> None:
    console.print(
        Panel(
            f"[bold]{result.model_name}[/bold] ({result.task}) — target: [cyan]{result.target}[/cyan]",
            title="[bold]Training completed[/bold]",
            border_style="green",
            expand=False,
        )
    )
    console.print()
    console.print(metrics_table(result.metrics_flat, title="Hold-out Performance"))
    console.print()
    cv_table = metrics_table(
        {"cv_mean": result.cv_scores["cv_mean"], "cv_std": result.cv_scores["cv_std"]},
        title=f"Cross-validation ({len(result.cv_scores['cv_scores'])} folds)",
    )
    cv_table.add_row(
        "Fold scores", " | ".join(str(s) for s in result.cv_scores["cv_scores"])
    )
    console.print(cv_table)
    console.print()
    rows = [
        ("Samples (train)", f"{result.n_train:,}"),
        ("Samples (validation)", f"{result.n_validation:,}"),
        ("Duration", f"{result.duration_seconds:.1f}s"),
        ("Features (transformed)", str(len(result.feature_names))),
    ]
    if model_path is not None:
        rows.append(("Model saved", str(model_path)))
    console.print(summary_table("Run Details", rows))
    if result.feature_importance:
        console.print()
        top = result.feature_importance[:10]
        table = Table(title="Top Feature Importance", box=box.SIMPLE_HEAD)
        table.add_column("Feature", style="cyan")
        table.add_column("Importance", justify="right")
        for name, score in top:
            table.add_row(name, f"{score:.4f}")
        console.print(table)


@app.command(help="Train a model and evaluate it on the validation set.")
@command
def train(
    ctx: typer.Context,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help=f"Model algorithm ({', '.join(available_models())})."),
    ] = None,
    target: Annotated[
        Optional[str], typer.Option("--target", help="Override the configured target column.")
    ] = None,
    task: Annotated[
        Optional[str],
        typer.Option("--task", help="Force task type: classification or regression (default: auto)."),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Where to save the model (default: models/<algorithm>.joblib)."),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite an existing model artifact.")
    ] = False,
    no_track: Annotated[
        bool, typer.Option("--no-track", help="Do not record this run in the experiment tracker.")
    ] = False,
    param: Annotated[
        Optional[list[str]],
        typer.Option("--param", help="Override a hyperparameter as key=value (repeatable)."),
    ] = None,
    project_dir: Annotated[Optional[Path], PROJECT_DIR_OPT] = None,
    config: Annotated[Optional[Path], CONFIG_OPT] = None,
    verbose: Annotated[bool, VERBOSE_OPT] = False,
) -> None:
    config, _ = _ctx_config(ctx)
    if task and task not in ("classification", "regression"):
        raise DataError(f"Invalid task '{task}'. Use 'classification' or 'regression'.")
    if model and model not in available_models():
        raise ModelError(
            f"Unknown model '{model}'. Available models: {', '.join(available_models())}."
        )

    train_path = config.resolve(config.data.train)
    validation_path = config.resolve(config.data.validation)
    df_train = _load_training_data(config, train_path)
    df_val = load_dataframe(validation_path) if validation_path.exists() else None
    if df_val is not None and config.data.id_column and config.data.id_column in df_val.columns:
        df_val = df_val.drop(columns=[config.data.id_column])

    model_params = _parse_params(param)
    result = train_model(
        df_train,
        config,
        algorithm=model,
        target=target,
        task=task,
        validation_df=df_val,
        model_params=model_params,
    )

    model_path = config.resolve(str(output)) if output else config.model_dir / f"{result.model_name}.joblib"
    check_can_write(model_path, overwrite or config.output.overwrite, description="model artifact")
    save_model(result, model_path, overwrite=True)

    if not no_track:
        with ExperimentTracker(config.experiments_db) as tracker:
            experiment_id = tracker.record(
                model=result.model_name,
                task=result.task,
                dataset=str(train_path),
                hyperparameters=model_params or config.model.params,
                metrics=result.metrics_flat,
                cv_scores=result.cv_scores,
                training_duration=result.duration_seconds,
                model_path=str(model_path),
                target=result.target,
                n_train=result.n_train,
                n_validation=result.n_validation,
            )

    _print_training_result(result, model_path)
    if not no_track:
        print_success(f"Experiment recorded: {experiment_id}")


def _parse_params(params: list[str] | None) -> dict[str, Any] | None:
    """Parse 'key=value' CLI overrides into a dict."""
    if not params:
        return None
    parsed: dict[str, Any] = {}
    for item in params:
        if "=" not in item:
            raise DataError(
                f"Invalid --param '{item}'. Expected key=value, e.g. --param n_estimators=500."
            )
        key, value = item.split("=", 1)
        value = value.strip()
        parsed[key] = _coerce_param(value)
    return parsed


def _coerce_param(value: str) -> Any:
    """Coerce a CLI param string to int/float/bool when possible."""
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


# ---------------------------------------------------------------------------
# Evaluation / comparison / prediction
# ---------------------------------------------------------------------------


@app.command(help="Evaluate a saved model on a test dataset.")
@command
def evaluate(
    ctx: typer.Context,
    model: Annotated[
        Optional[Path],
        typer.Option("--model", "-m", help="Path to a saved model artifact (default: the trained model)."),
    ] = None,
    input: Annotated[
        Optional[Path],
        typer.Option("--input", "-i", help="Test dataset (default: the configured test split)."),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Where to write metrics JSON (default: reports/reports/evaluation_<model>.json)."),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite existing output files.")
    ] = False,
    project_dir: Annotated[Optional[Path], PROJECT_DIR_OPT] = None,
    config: Annotated[Optional[Path], CONFIG_OPT] = None,
    verbose: Annotated[bool, VERBOSE_OPT] = False,
) -> None:
    config, _ = _ctx_config(ctx)
    model_path = _resolve_model_path(config, model)
    pipeline, metadata = load_model(model_path)

    in_path = config.resolve(str(input)) if input else config.resolve(config.data.test)
    df = load_dataframe(in_path)
    if metadata.target not in df.columns:
        raise DataError(
            f"Evaluation dataset '{in_path}' is missing the target column "
            f"'{metadata.target}' required by model '{metadata.model_name}'."
        )

    y_true_raw = df[metadata.target]
    X = df.drop(columns=[metadata.target])
    y_pred, y_proba = predict_with_pipeline(pipeline, X, metadata.task)

    if metadata.task == "classification":
        label_map = {label: i for i, label in enumerate(metadata.classes or [])}
        y_true = [label_map.get(v, -1) for v in y_true_raw]
        try:
            metrics = compute_metrics(y_true, y_pred, metadata.task, y_proba)
        except Exception:
            metrics = compute_metrics(y_true, y_pred, metadata.task, None)
    else:
        metrics = compute_metrics(y_true_raw, y_pred, metadata.task)

    importance = extract_feature_importance(pipeline, metadata.feature_names)
    y_true_encoded = (
        np.asarray(y_true) if metadata.task == "classification" else y_true_raw.to_numpy()
    )
    figures = generate_figures(
        df,
        config.figure_dir,
        target=metadata.target,
        metrics=metrics,
        y_true=y_true_encoded,
        y_pred=y_pred,
        y_proba=y_proba,
        feature_importance=importance,
        task=metadata.task,
    )

    out_path = (
        config.resolve(str(output))
        if output
        else config.resolve(f"reports/reports/evaluation_{model_path.stem}.json")
    )
    check_can_write(out_path, overwrite or config.output.overwrite, description="metrics file")
    save_json(
        {
            "model": model_path.stem,
            "algorithm": metadata.model_name,
            "task": metadata.task,
            "target": metadata.target,
            "dataset": str(in_path),
            "metrics": metrics,
            "figures": figures,
        },
        out_path,
        overwrite=True,
    )

    console.print(
        Panel(
            f"[bold]{metadata.model_name}[/bold] evaluated on [cyan]{in_path}[/cyan]",
            title="[bold]Evaluation[/bold]",
            border_style="green",
            expand=False,
        )
    )
    console.print(metrics_table({k: v for k, v in metrics.items() if isinstance(v, (int, float))}))
    print_success(f"Metrics written to '{out_path}'.")
    print_success(f"Generated {len(figures)} figure(s).")


@app.command(help="Train several models and compare their performance.")
@command
def compare(
    ctx: typer.Context,
    models: Annotated[
        Optional[str],
        typer.Option("--models", "-m", help="Comma-separated algorithms to compare (default: 3 core models)."),
    ] = None,
    target: Annotated[
        Optional[str], typer.Option("--target", help="Override the configured target column.")
    ] = None,
    task: Annotated[
        Optional[str], typer.Option("--task", help="Force task type: classification or regression.")
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Where to write the comparison JSON."),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite existing outputs.")
    ] = False,
    project_dir: Annotated[Optional[Path], PROJECT_DIR_OPT] = None,
    config: Annotated[Optional[Path], CONFIG_OPT] = None,
    verbose: Annotated[bool, VERBOSE_OPT] = False,
) -> None:
    config, _ = _ctx_config(ctx)
    train_path = config.resolve(config.data.train)
    validation_path = config.resolve(config.data.validation)
    df_train = _load_training_data(config, train_path)
    df_val = load_dataframe(validation_path) if validation_path.exists() else None

    target_name = target or config.data.target
    _, inferred_task = validate_target(df_train, target_name)
    effective_task = task or inferred_task

    if models:
        model_names = [m.strip() for m in models.split(",") if m.strip()]
    else:
        model_names = (
            ["logistic_regression", "random_forest", "gradient_boosting"]
            if effective_task == "classification"
            else ["ridge", "random_forest", "gradient_boosting"]
        )
    for name in model_names:
        if name not in available_models(effective_task):
            raise ModelError(
                f"Model '{name}' is not available for {effective_task}. "
                f"Available: {', '.join(available_models(effective_task))}."
            )

    results: list[TrainingResult] = []
    with console.status(f"Training {len(model_names)} model(s)...", spinner="dots"):
        for name in model_names:
            print_info(f"Training {name}...")
            result = train_model(
                df_train,
                config,
                algorithm=name,
                target=target_name,
                task=effective_task,
                validation_df=df_val,
            )
            results.append(result)
            with ExperimentTracker(config.experiments_db) as tracker:
                tracker.record(
                    model=name,
                    task=effective_task,
                    dataset=str(train_path),
                    hyperparameters=config.model.params,
                    metrics=result.metrics_flat,
                    cv_scores=result.cv_scores,
                    training_duration=result.duration_seconds,
                    target=target_name,
                    n_train=result.n_train,
                    n_validation=result.n_validation,
                    note="compare",
                )

    primary = "accuracy" if effective_task == "classification" else "r2"
    secondary = "f1" if effective_task == "classification" else "rmse"
    table = Table(title="Model Comparison", box=box.ROUNDED)
    table.add_column("Model", style="cyan")
    table.add_column(METRIC_LABELS.get(primary, primary), justify="right")
    table.add_column(METRIC_LABELS.get(secondary, secondary), justify="right")
    table.add_column("CV (mean)", justify="right")
    table.add_column("Duration (s)", justify="right")
    for result in sorted(results, key=lambda r: r.metrics_flat.get(primary, -1), reverse=True):
        table.add_row(
            result.model_name,
            f"{result.metrics_flat.get(primary, 0):.4f}",
            f"{result.metrics_flat.get(secondary, 0):.4f}",
            f"{result.cv_scores['cv_mean']:.4f}",
            f"{result.duration_seconds:.1f}",
        )
    console.print(table)

    best = max(results, key=lambda r: r.metrics_flat.get(primary, -1))
    print_success(f"Best model: {best.model_name} ({primary}={best.metrics_flat.get(primary):.4f}).")

    out_path = (
        config.resolve(str(output))
        if output
        else config.resolve("reports/reports/comparison.json")
    )
    check_can_write(out_path, overwrite or config.output.overwrite, description="comparison file")
    save_json(
        {
            "task": effective_task,
            "target": target_name,
            "models": [
                {
                    "model": r.model_name,
                    "metrics": r.metrics_flat,
                    "cv_scores": r.cv_scores,
                    "duration_seconds": r.duration_seconds,
                }
                for r in results
            ],
        },
        out_path,
        overwrite=True,
    )
    print_success(f"Comparison written to '{out_path}'.")


@app.command(help="Generate predictions for new data with a saved model.")
@command
def predict(
    ctx: typer.Context,
    input: Annotated[Path, typer.Option("--input", "-i", help="Dataset to score.")],
    model: Annotated[
        Optional[Path],
        typer.Option("--model", "-m", help="Path to a saved model artifact (default: the trained model)."),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Where to write predictions (default: predictions.csv)."),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite an existing predictions file.")
    ] = False,
    project_dir: Annotated[Optional[Path], PROJECT_DIR_OPT] = None,
    config: Annotated[Optional[Path], CONFIG_OPT] = None,
    verbose: Annotated[bool, VERBOSE_OPT] = False,
) -> None:
    config, _ = _ctx_config(ctx)
    in_path = config.resolve(str(input))
    df = load_dataframe(in_path)

    model_path = _resolve_model_path(config, model)
    pipeline, metadata = load_model(model_path)

    X = df.copy()
    if metadata.target in X.columns:
        X = X.drop(columns=[metadata.target])
    if config.data.id_column and config.data.id_column in X.columns:
        X = X.drop(columns=[config.data.id_column])

    y_pred, y_proba = predict_with_pipeline(pipeline, X, metadata.task)

    out_df = pd.DataFrame({"prediction": y_pred})
    if metadata.task == "classification" and y_proba is not None:
        for idx, cls in enumerate(metadata.classes or []):
            out_df[f"probability_{cls}"] = y_proba[:, idx]

    id_col = config.data.id_column
    if id_col and id_col in df.columns:
        out_df.insert(0, id_col, df[id_col].values)
    else:
        out_df.insert(0, "id", df.index)

    out_path = config.resolve(str(output)) if output else config.resolve("predictions.csv")
    check_can_write(out_path, overwrite or config.output.overwrite, description="predictions file")
    out_df.to_csv(out_path, index=False)

    console.print(
        Panel(
            f"[bold]{metadata.model_name}[/bold] scored [cyan]{len(df):,}[/cyan] rows",
            title="[bold]Predictions[/bold]",
            border_style="green",
            expand=False,
        )
    )
    print_success(f"Predictions written to '{out_path}'.")
    console.print(out_df.head(10).to_string(index=False))


@app.command(help="Generate a Markdown report with figures.")
@command
def report(
    ctx: typer.Context,
    input: Annotated[
        Optional[Path],
        typer.Option("--input", "-i", help="Dataset to summarize (default: the training split)."),
    ] = None,
    model: Annotated[
        Optional[Path],
        typer.Option("--model", "-m", help="Path to a saved model artifact (default: the trained model)."),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Where to write the report (default: reports/reports/report.md)."),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite an existing report.")
    ] = False,
    project_dir: Annotated[Optional[Path], PROJECT_DIR_OPT] = None,
    config: Annotated[Optional[Path], CONFIG_OPT] = None,
    verbose: Annotated[bool, VERBOSE_OPT] = False,
) -> None:
    config, _ = _ctx_config(ctx)
    data_path = config.resolve(str(input)) if input else config.resolve(config.data.train)
    if not data_path.exists():
        raise DataError(f"Dataset not found: '{data_path}'. Pass --input to point at a dataset.")

    model_path: Path | None = None
    if model is not None or any(config.model_dir.glob("*.joblib")):
        model_path = _resolve_model_path(config, model)

    out_path = config.resolve(str(output)) if output else config.resolve("reports/reports/report.md")
    generate_report(
        config,
        data_path=data_path,
        model_path=model_path,
        output_path=out_path,
        overwrite=overwrite or config.output.overwrite,
    )
    print_success(f"Report written to '{out_path}'.")


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


@experiments_app.command("list", help="List recorded experiments.")
@command
def experiments_list(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Maximum number of experiments.")] = 50,
    project_dir: Annotated[Optional[Path], PROJECT_DIR_OPT] = None,
    config: Annotated[Optional[Path], CONFIG_OPT] = None,
    verbose: Annotated[bool, VERBOSE_OPT] = False,
) -> None:
    config, _ = _ctx_config(ctx)
    with ExperimentTracker(config.experiments_db) as tracker:
        experiments = tracker.list(limit=limit)

    if not experiments:
        print_info("No experiments recorded yet. Run 'dscli train' to create one.")
        return

    table = Table(title=f"Experiments ({len(experiments)})", box=box.ROUNDED)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Timestamp", style="dim", no_wrap=True)
    table.add_column("Model")
    table.add_column("Task", justify="center")
    table.add_column("Target")
    table.add_column("Primary metric", justify="right")
    table.add_column("Duration (s)", justify="right")
    for exp in experiments:
        primary = exp.primary_metric
        metric_str = f"{primary[1]:.4f}" if primary else "—"
        table.add_row(
            exp.id,
            exp.timestamp[:16].replace("T", " "),
            exp.model,
            exp.task,
            exp.target or "—",
            metric_str,
            f"{exp.training_duration:.1f}" if exp.training_duration else "—",
        )
    # Use a dedicated wide console so full values render even in narrow
    # terminals instead of being ellipsized (e.g. 'random_…').
    from rich.console import Console as _Console

    _Console(width=120, highlight=False).print(table)
    print_info("Run 'dscli experiments show <id>' for full details.")


@experiments_app.command("show", help="Show full details of an experiment.")
@command
def experiments_show(
    ctx: typer.Context,
    experiment_id: Annotated[str, typer.Argument(help="Experiment id.")],
    project_dir: Annotated[Optional[Path], PROJECT_DIR_OPT] = None,
    config: Annotated[Optional[Path], CONFIG_OPT] = None,
    verbose: Annotated[bool, VERBOSE_OPT] = False,
) -> None:
    config, _ = _ctx_config(ctx)
    with ExperimentTracker(config.experiments_db) as tracker:
        exp = tracker.get(experiment_id)

    rows = [
        ("ID", exp.id),
        ("Timestamp", exp.timestamp),
        ("Model", exp.model),
        ("Task", exp.task),
        ("Target", exp.target or "—"),
        ("Dataset", exp.dataset or "—"),
        ("Training samples", str(exp.n_train) if exp.n_train is not None else "—"),
        ("Validation samples", str(exp.n_validation) if exp.n_validation is not None else "—"),
        ("Duration (s)", f"{exp.training_duration:.1f}" if exp.training_duration else "—"),
        ("Model path", exp.model_path or "—"),
        ("Note", exp.note or "—"),
    ]
    console.print(summary_table(f"Experiment {experiment_id}", rows))
    if exp.metrics:
        console.print()
        console.print(metrics_table(exp.metrics, title="Metrics"))
    if exp.hyperparameters:
        console.print()
        hyper_table = Table(title="Hyperparameters", box=box.SIMPLE_HEAD)
        hyper_table.add_column("Parameter", style="cyan")
        hyper_table.add_column("Value")
        for key, value in exp.hyperparameters.items():
            hyper_table.add_row(key, str(value))
        console.print(hyper_table)


@experiments_app.command("delete", help="Delete an experiment.")
@command
def experiments_delete(
    ctx: typer.Context,
    experiment_id: Annotated[str, typer.Argument(help="Experiment id.")],
    project_dir: Annotated[Optional[Path], PROJECT_DIR_OPT] = None,
    config: Annotated[Optional[Path], CONFIG_OPT] = None,
    verbose: Annotated[bool, VERBOSE_OPT] = False,
) -> None:
    config, _ = _ctx_config(ctx)
    with ExperimentTracker(config.experiments_db) as tracker:
        tracker.delete(experiment_id)
    print_success(f"Experiment '{experiment_id}' deleted.")


@experiments_app.command("export", help="Export all experiments to JSON.")
@command
def experiments_export(
    ctx: typer.Context,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output file (default: experiments.json).")
    ] = Path("experiments.json"),
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite an existing export.")
    ] = False,
    project_dir: Annotated[Optional[Path], PROJECT_DIR_OPT] = None,
    config: Annotated[Optional[Path], CONFIG_OPT] = None,
    verbose: Annotated[bool, VERBOSE_OPT] = False,
) -> None:
    config, _ = _ctx_config(ctx)
    out_path = config.resolve(str(output))
    with ExperimentTracker(config.experiments_db) as tracker:
        written = tracker.export(out_path)
        count = tracker.count()
    print_success(f"Exported {count} experiment(s) to '{written}'.")


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------


@app.command(help="Generate a synthetic demo dataset.")
@command
def demo_data(
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output file (default: data/raw/demo.csv).")
    ] = Path("data/raw/demo.csv"),
    rows: Annotated[int, typer.Option("--rows", help="Number of rows.")] = 2000,
    task: Annotated[
        str, typer.Option("--task", help="Task type: classification or regression.")
    ] = "classification",
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite an existing file.")
    ] = False,
    project_dir: Annotated[Optional[Path], PROJECT_DIR_OPT] = None,
    config: Annotated[Optional[Path], CONFIG_OPT] = None,
    verbose: Annotated[bool, VERBOSE_OPT] = False,
) -> None:
    df = generate_demo_dataset(rows=rows, task=task, seed=seed)
    out_path = Path(output)
    check_can_write(out_path, overwrite, description="dataset")
    save_dataframe(df, out_path, overwrite=True)
    print_success(
        f"Wrote {len(df):,} rows × {len(df.columns)} columns to '{out_path}' "
        f"(target: {('churn' if task == 'classification' else 'price')})."
    )


if __name__ == "__main__":
    app()
