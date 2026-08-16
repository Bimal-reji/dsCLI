"""Tests for the configuration module."""

from __future__ import annotations

import pytest
import yaml

from dscli.config import Config, deep_merge, find_project_root
from dscli.errors import ConfigError


def test_defaults_are_sane():
    config = Config.from_dict({}, project_root=".")
    assert config.project.name == "my-project"
    assert config.data.target is None
    assert config.training.test_size == 0.2
    assert config.features.scaler == "standard"
    assert config.model.algorithm == "random_forest"


def test_deep_merge_nested():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"c": 9}, "e": 4}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 1, "c": 9}, "d": 3, "e": 4}
    # The original must not be mutated.
    assert base["a"]["c"] == 2


def test_from_dict_overrides_defaults():
    config = Config.from_dict(
        {"data": {"target": "price"}, "training": {"cv_folds": 10}},
        project_root=".",
    )
    assert config.data.target == "price"
    assert config.training.cv_folds == 10
    assert config.training.random_state == 42  # untouched default


def test_invalid_choice_raises():
    with pytest.raises(ConfigError, match="missing_strategy"):
        Config.from_dict({"cleaning": {"missing_strategy": "bogus"}}, project_root=".")


def test_unknown_key_raises():
    with pytest.raises(ConfigError, match="Unknown key"):
        Config.from_dict({"data": {"not_a_key": 1}}, project_root=".")


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        Config.load(tmp_path / "nope.yaml")


def test_load_and_dump_roundtrip(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"project": {"name": "demo"}, "model": {"algorithm": "ridge"}}),
        encoding="utf-8",
    )
    config = Config.load(path, project_root=tmp_path)
    assert config.project.name == "demo"
    assert config.model.algorithm == "ridge"
    assert config.project_root == tmp_path.resolve()

    out = tmp_path / "out.yaml"
    config.dump(out)
    reloaded = Config.load(out, project_root=tmp_path)
    assert reloaded.to_dict() == config.to_dict()


def test_resolve_paths_relative_to_project_root(tmp_path):
    config = Config.from_dict({}, project_root=tmp_path)
    assert config.resolve("data/raw") == (tmp_path / "data/raw").resolve()
    # Absolute paths pass through untouched.
    abs_path = tmp_path / "some" / "absolute" / "path"
    assert config.resolve(str(abs_path)) == abs_path


def test_with_overrides_returns_new_config():
    config = Config.from_dict({}, project_root=".")
    updated = config.with_overrides({"model": {"algorithm": "ridge"}})
    assert config.model.algorithm == "random_forest"  # original untouched
    assert updated.model.algorithm == "ridge"


def test_find_project_root_finds_marker(tmp_path):
    root = tmp_path / "deep" / "project"
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "config.yaml").write_text("project:\n  name: x\n", encoding="utf-8")
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    assert find_project_root(nested) == root
    assert find_project_root(tmp_path) is None
