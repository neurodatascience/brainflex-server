from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from fedbiomed.common.dataset import Dataset
from fedbiomed.common.training_plans import BaseTrainingPlan


@pytest.fixture
def model_args(model_args: dict) -> dict:
    return model_args | {"target_column": "nb:Sex", "n_classes": 2}


@pytest.fixture
def plan_cls(
    load_training_plan: Callable[[str, str], type[BaseTrainingPlan]],
) -> type[BaseTrainingPlan]:
    return load_training_plan("tabular_classifier", "TabularClassifierTrainingPlan")


class TestTabularClassifierTrainingPlan:
    def test_training_data(
        self,
        plan_cls: type[BaseTrainingPlan],
        fpath_csv: Path,
        model_args: dict,
        load_dataset: Callable[[BaseTrainingPlan, dict, Path], Dataset],
    ):
        dataset = load_dataset(plan_cls(), model_args, fpath_csv)
        samples = [dataset[i] for i in range(len(dataset))]
        np.testing.assert_allclose(
            [x for x, _ in samples], [[1.0, -1.0], [0.0, 0.0], [-1.0, 1.0]]
        )
        np.testing.assert_allclose([y for _, y in samples], [[0], [1], [1]])

    @pytest.mark.parametrize(
        "extra_args",
        [
            {"n_features": 3},
            {"feature_columns": ["Left-Hippocampus"]},
            {"feature_mean": [4000.0]},
            {"feature_std": [100.0, 0.5, 1.0]},
        ],
    )
    def test_length_mismatch(
        self,
        plan_cls: type[BaseTrainingPlan],
        fpath_csv: Path,
        model_args: dict,
        load_dataset: Callable[[BaseTrainingPlan, dict, Path], Dataset],
        extra_args: dict,
    ):
        with pytest.raises(ValueError, match="must all have length"):
            load_dataset(plan_cls(), model_args | extra_args, fpath_csv)

    def test_from_source(
        self,
        plan_cls: type[BaseTrainingPlan],
        fpath_csv: Path,
        model_args: dict,
        load_dataset: Callable[[BaseTrainingPlan, dict, Path], Dataset],
        from_source: Callable[[BaseTrainingPlan], type[BaseTrainingPlan]],
    ):
        dataset = load_dataset(from_source(plan_cls())(), model_args, fpath_csv)
        assert len(dataset) == 3

    @pytest.mark.parametrize(
        "optimizer_args,expected_modules,expected_regularizers",
        [
            ({"lr": 0.01}, [], []),
            ({"lr": 0.01, "scaffold": True}, [("scaffold-client", {})], []),
            (
                {"lr": 0.01, "fedprox_mu": 2.0},
                [],
                [("fedprox", {"alpha": 2.0})],
            ),
        ],
    )
    def test_optimizer(
        self,
        plan_cls: type[BaseTrainingPlan],
        model_args: dict,
        optimizer_config: Callable[[BaseTrainingPlan, dict, dict], dict | None],
        optimizer_args: dict,
        expected_modules: list,
        expected_regularizers: list,
    ):
        config = optimizer_config(plan_cls(), model_args, optimizer_args)
        assert config["lrate"] == 0.01
        assert config["modules"] == expected_modules
        assert config["regularizers"] == expected_regularizers

    def test_no_optimizer_args(
        self,
        plan_cls: type[BaseTrainingPlan],
        model_args: dict,
        optimizer_config: Callable[[BaseTrainingPlan, dict, dict], dict | None],
    ):
        assert optimizer_config(plan_cls(), model_args, {}) is None
