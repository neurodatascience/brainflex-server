import dataclasses
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture
def module(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("_io_utils")


@pytest.fixture
def path_helper(module: ModuleType, tmp_path: Path) -> object:
    return module.PathHelper(
        output_dir=tmp_path,
        date_tag="2026_01_01",
        n_rounds=10,
        batch_size=32,
        n_updates=50,
        learning_rate=0.001,
        test_ratio=0.2,
        aggregator="scaffold",
        fedprox_mu=2.0,
        alpha=0.1,
        secagg=True,
        cv=True,
    )


class TestPathHelper:
    def test_get_fpath_out(self, path_helper: object, tmp_path: Path):
        assert path_helper.get_fpath_out("age", "results") == (
            tmp_path
            / "2026_01_01"
            / "age"
            / (
                "results-scaffold-10rounds-50updates-32-0.001lr-0.2test-0.1alpha"
                "-secagg-cv.json"
            )
        )

    def test_no_secagg_no_cv(self, path_helper: object):
        path_helper = dataclasses.replace(path_helper, secagg=False, cv=False)
        fpath = path_helper.get_fpath_out("age", "results")
        assert fpath.stem.endswith("-0.1alpha-no_secagg-no_cv")

    def test_fedprox(self, path_helper: object):
        path_helper = dataclasses.replace(path_helper, aggregator="fedprox")
        fpath = path_helper.get_fpath_out("age", "results")
        assert fpath.name.startswith("results-fedprox_2.0mu-")

    def test_extension(self, path_helper: object, tmp_path: Path):
        fpath = path_helper.get_fpath_out("sex", "model", ".pkl")
        assert fpath.parent == tmp_path / "2026_01_01" / "sex"
        assert fpath.suffix == ".pkl"

    def test_held_out_node(self, path_helper: object):
        fpath = path_helper.get_fpath_out(
            "age", "model", ".pkl", held_out_node="NODE_1"
        )
        assert fpath.name.endswith("-0.1alpha-secagg-cv-heldout_NODE_1.pkl")


class TestSerialize:
    @pytest.mark.parametrize(
        "obj,expected",
        [
            (Enum("Color", "RED").RED, "RED"),
            (Path("a/b"), "a/b"),
            (np.array([1.0, 2.0]), [1.0, 2.0]),
            (np.float64(1.5), 1.5),
            ({"b", "a"}, ["a", "b"]),
        ],
    )
    def test_serialize(self, module: ModuleType, obj: object, expected: object):
        assert module.serialize(obj) == expected

    def test_not_serializable(self, module: ModuleType):
        with pytest.raises(TypeError, match="not serializable"):
            module.serialize(object())
