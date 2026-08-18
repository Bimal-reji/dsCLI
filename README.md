# dscli — Data Science CLI

A production-quality command-line interface for managing the complete
ML/data-science workflow from the terminal: initialize a project, load and
clean data, explore it, engineer features, split, train, evaluate, compare,,
predict, report, and track experiments — all without leaving your shell.

Built with **Typer**, **Rich**, **pandas**, **scikit-learn**, **matplotlib**,
and **seaborn**. Works on Windows, macOS, and Linux.

```text
╭──────────────────────────────────────╮
│        Data Science Pipeline         │
╰──────────────────────────────────────╯

✓ Dataset loaded
✓ 10,542 rows
✓ 24 features
✓ Missing values: 1.8%

Training Random Forest...
━━━━━━━━━━━━━━━━━━━━━━━ 100%

✓ Training completed

Model Performance
┌────────────┬────────┐
│ Metric     │ Score  │
├────────────┼────────┤
│ Accuracy   │ 0.942  │
│ Precision  │ 0.931  │
│ Recall     │ 0.918  │
│ F1 Score   │ 0.924  │
└────────────┴────────┘

✓ Model saved to models/random_forest.joblib
```

---

## Quick start

```bash
# 1. Install
pip install -e .

# 2. Create a project with a ready-to-use demo dataset
dscli init my-project --demo
cd my-project

# 3. Run the full pipeline
dscli data info data/raw/train.csv   # inspect
dscli data validate                  # check quality
dscli data clean                     # clean -> data/interim/train.csv
dscli split                          # train/validation/test splits
dscli train                          # train + evaluate (random forest)
dscli evaluate                       # evaluate on the test split
dscli predict --input data/raw/demo.csv --output predictions.csv
dscli report                         # Markdown report + figures
dscli experiments list               # every run is tracked
```

---

## Installation

Requires **Python 3.11+**.

```bash
# From the project root (editable install; includes dev dependencies):
pip install -e ".[dev]"

# Optional model libraries (not required — used automatically when installed):
pip install -e ".[xgboost]"
pip install -e ".[lightgbm]"
```

The `dscli` command is exposed via the `[project.scripts]` entry point in
`pyproject.toml`, so no wrapper scripts are needed.

---

## CLI reference

Global options (accepted by every command):

| Option | Description |
| --- | --- |
| `--project-dir PATH` | Project root (default: nearest project found upward from cwd) |
| `--config PATH` | Path to a YAML config file |
| `--verbose` / `-v` | Debug logging + full tracebacks |

Every command supports `--help` (`-h`).

### Project scaffolding

| Command | Description |
| --- | --- |
| `dscli init [NAME] [--demo] [--force]` | Scaffold a project (optionally with demo data) |
| `dscli status` | Show project status, datasets, models, experiments |
| `dscli config show` | Print the effective configuration as YAML |
| `dscli demo-data [--rows N] [--task classification\|regression]` | Generate a synthetic dataset |

### Data

| Command | Description |
| --- | --- |
| `dscli data info <path>` | Summary: rows, columns, dtypes, missing values |
| `dscli data validate [--input] [--target]` | Validate columns, missingness, duplicates, target |
| `dscli data clean [--input] [--output] [--overwrite]` | Dedupe, impute, cap outliers |

### Exploration & features

| Command | Description |
| --- | --- |
| `dscli eda [--input]` | Numeric summary + correlation/distribution figures |
| `dscli features build [--input] [--output]` | Feature matrix + reusable preprocessor |
| `dscli split [--input]` | Train/validation/test splits (test_size, validation_size) |

### Modeling

| Command | Description |
| --- | --- |
| `dscli train [--model NAME] [--target] [--task] [--param k=v ...] [--no-track]` | Train + evaluate, save artifact, record experiment |
| `dscli evaluate [--model] [--input] [--output]` | Evaluate a saved model on test data; writes metrics JSON + figures |
| `dscli compare [--models a,b,c]` | Train several models and rank them |
| `dscli predict --input <path> [--model] [--output]` | Score new data; writes CSV with probabilities |
| `dscli report [--model] [--output]` | Markdown report with figures |

### Experiments

| Command | Description |
| --- | --- |
| `dscli experiments list [--limit N]` | List runs (newest first) |
| `dscli experiments show <id>` | Full details: metrics, hyperparameters, paths |
| `dscli experiments delete <id>` | Remove a run |
| `dscli experiments export [--output]` | Export all runs to JSON |

### Exit codes

- `0` — success
- `1` — any error (missing files, invalid config, failed training, ...).
  Errors are rendered as readable messages, never raw tracebacks (use
  `--verbose` to see the full traceback).

---

## Example workflow

```bash
dscli init customer-churn --demo
cd customer-churn

dscli data info data/raw/train.csv
dscli data validate
dscli data clean                          # -> data/interim/train.csv
dscli split                               # -> data/processed/{train,validation,test}.csv
dscli eda                                 # figures in reports/figures/

# Try a few models, then compare:
dscli train --model logistic_regression
dscli train --model random_forest
dscli train --model gradient_boosting
dscli compare

# The best model (or any saved one) can be evaluated and used:
dscli evaluate --model models/random_forest.joblib
dscli predict --input data/raw/new_customers.csv --output predictions.csv

# Document everything:
dscli report
dscli experiments list
```

Every `train`/`compare` run is recorded in `experiments.db` (SQLite) with its
model, hyperparameters, dataset, metrics, cross-validation scores, duration,
and artifact path.

---

## Configuration

All paths and hyperparameters live in `configs/config.yaml`; nothing is
hard-coded in the CLI. Relative paths are resolved against the project root,
so commands work from any subdirectory.

```yaml
project:
  name: customer_churn

data:
  train: data/processed/train.csv
  test: data/processed/test.csv
  target: churn
  id_column: customer_id

model:
  algorithm: random_forest
  params: {}            # e.g. {n_estimators: 500}

training:
  test_size: 0.2
  validation_size: 0.1
  cv_folds: 5
  random_state: 42

output:
  model_dir: models
  report_dir: reports
  overwrite: false
```

CLI arguments override configuration values (e.g. `dscli train --model xgboost
--param n_estimators=500 --target churn`). `dscli config show` prints the
effective merged configuration.

---

## Project structure

```text
my-project/
├── data/
│   ├── raw/           # untouched source data
│   ├── interim/       # cleaned data
│   ├── processed/     # splits and feature matrices
│   └── external/      # third-party data
├── notebooks/
├── models/            # saved .joblib pipelines (+ .json metadata sidecars)
├── reports/
│   ├── figures/       # generated PNGs
│   └── reports/       # evaluation metrics, comparison, report.md
├── configs/
│   └── config.yaml
├── logs/              # rotating dscli.log
├── experiments.db     # SQLite experiment tracker
└── .gitignore
```

The package itself is a **src-layout** Python project:

```text
src/dscli/
├── cli.py              # Typer CLI: parsing, validation, rendering only
├── config.py           # dataclass config, YAML loading, deep merge
├── errors.py           # exception hierarchy (all user-facing errors)
├── report.py           # Markdown report generation
├── data/               # loader, validator, cleaner, demo data
├── features/           # reusable preprocessing pipeline builder
├── models/             # registry, trainer, persistence
├── evaluation/         # metrics + feature importance
├── visualization/      # matplotlib/seaborn figures
├── experiments/        # SQLite experiment tracker
└── utils/              # logging, console, file I/O
```

### Architecture

The CLI layer (`cli.py`) only parses arguments, validates them, and renders
results. All business logic lives in the `data/`, `features/`, `models/`,
`evaluation/`, and `experiments/` modules, which are plain functions and
dataclasses — no global mutable state, usable from notebooks or other tools.

- **Preprocessing is a single reusable artifact.** The numeric/categorical
  `ColumnTransformer` is fitted on training data only and saved *inside* the
  model pipeline, so `train`, `evaluate`, and `predict` apply identical
  transformations (imputation, scaling, encoding) to every dataset.
- **Task type is auto-detected.** A numeric target with few unique values or
  any non-numeric target → classification; otherwise → regression. Override
  with `--task`.
- **No silent overwrites.** Saving to an existing path fails with a clear
  message unless `--overwrite` (or `output.overwrite: true`) is given.
- **Errors are data-scientist friendly.** Missing files, missing columns,
  invalid configs, unknown models, and failed training all produce a
  one-paragraph explanation and exit code 1 instead of a traceback.

### Supported models

| Algorithm | Classification | Regression |
| --- | --- | --- |
| `logistic_regression` | ✅ | — |
| `ridge` | — | ✅ |
| `random_forest` | ✅ | ✅ |
| `gradient_boosting` | ✅ | ✅ |
| `xgboost` *(optional)* | ✅ | ✅ |
| `lightgbm` *(optional)* | ✅ | ✅ |

Metrics are chosen automatically: **classification** → accuracy, precision,
recall, F1, ROC-AUC, confusion matrix; **regression** → MAE, MSE, RMSE, R².

---

## Development

```bash
pip install -e ".[dev]"
pytest                 # run the test suite
pytest --cov=src/dscli # with coverage
```

The test suite covers configuration loading, data validation/cleaning,
feature building, model training (classification + regression), evaluation,
persistence round-trips, the experiment tracker, and end-to-end CLI workflows
via Typer's `CliRunner`.

### Style

- Type hints throughout, PEP 8, docstrings on public functions/classes.
- Structured logging via `logging` (rotating file + optional Rich console
  handler); the CLI renders user-facing output with Rich, never `print`.
- Modules stay small; new commands wire into `cli.py` as thin wrappers over
  the business-logic modules.

---

## License

MIT
