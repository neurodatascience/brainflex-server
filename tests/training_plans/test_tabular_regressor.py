from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fedbiomed.common.dataset import Dataset
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import BaseTrainingPlan


@pytest.fixture
def model_args(model_args: dict) -> dict:
    return model_args | {"target_column": "nb:Age"}


@pytest.fixture
def plan_cls(
    load_training_plan: Callable[[str, str], type[BaseTrainingPlan]],
) -> type[BaseTrainingPlan]:
    return load_training_plan("tabular_regressor", "TabularRegressorTrainingPlan")


class TestTabularRegressorTrainingPlan:
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
        np.testing.assert_allclose([y for _, y in samples], [[20.0], [30.0], [40.0]])

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

    def test_intercept_init(self, plan_cls: type[BaseTrainingPlan], model_args: dict):
        plan = plan_cls()
        plan.post_init(
            model_args | {"target_mean": 30.0},
            TrainingArgs({}, only_required=False),
        )
        params = plan.get_model_params()
        np.testing.assert_array_equal(params["intercept_"], [30.0])
        np.testing.assert_array_equal(params["coef_"], [0.0, 0.0])

    def test_intercept_default(
        self, plan_cls: type[BaseTrainingPlan], model_args: dict
    ):
        plan = plan_cls()
        plan.post_init(model_args, TrainingArgs({}, only_required=False))
        np.testing.assert_array_equal(plan.get_model_params()["intercept_"], [0.0])

    def test_no_training_data(self, plan_cls: type[BaseTrainingPlan]):
        plan = plan_cls()
        test_data_loader = MagicMock()
        plan.set_data_loaders(None, test_data_loader)
        assert plan.training_data_loader is None
        assert plan.testing_data_loader is test_data_loader

    @pytest.mark.parametrize("use_source", [False, True])
    def test_testing_step(
        self,
        plan_cls: type[BaseTrainingPlan],
        model_args: dict,
        from_source: Callable[[BaseTrainingPlan], type[BaseTrainingPlan]],
        use_source: bool,
    ):
        plan = plan_cls()
        if use_source:
            plan = from_source(plan)()
        plan.post_init(
            model_args | {"metrics": list(plan_cls.METRICS)},
            TrainingArgs({}, only_required=False),
        )
        plan.set_model_params(
            {"coef_": np.array([1.0, 0.0]), "intercept_": np.array([0.0])}
        )
        # predictions: [1, 2, 3, 4]
        data = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])
        target = np.array([[1.0], [2.0], [4.0], [5.0]])
        metrics = plan.testing_step(data, target)
        assert metrics.keys() == {"MEAN_SQUARE_ERROR", "MEAN_ABSOLUTE_ERROR", "R2"}
        np.testing.assert_allclose(metrics["MEAN_SQUARE_ERROR"], 0.5)
        np.testing.assert_allclose(metrics["MEAN_ABSOLUTE_ERROR"], 0.5)
        # 1 - 2 / 10
        np.testing.assert_allclose(metrics["R2"], 0.8)

    def test_testing_step_metrics_arg(
        self, plan_cls: type[BaseTrainingPlan], model_args: dict
    ):
        plan = plan_cls()
        plan.post_init(
            model_args | {"metrics": ["MEAN_ABSOLUTE_ERROR"]},
            TrainingArgs({}, only_required=False),
        )
        plan.set_model_params(
            {"coef_": np.array([1.0, 0.0]), "intercept_": np.array([0.0])}
        )
        metrics = plan.testing_step(np.array([[1.0, 5.0]]), np.array([[3.0]]))
        assert metrics == {"MEAN_ABSOLUTE_ERROR": 2.0}

    def test_unknown_metric(self, plan_cls: type[BaseTrainingPlan], model_args: dict):
        with pytest.raises(ValueError, match="Unknown metrics"):
            plan_cls().post_init(
                model_args | {"metrics": ["R2", "RMSE"]},
                TrainingArgs({}, only_required=False),
            )

    def test_testing_step_default(
        self, plan_cls: type[BaseTrainingPlan], model_args: dict
    ):
        plan = plan_cls()
        plan.post_init(model_args, TrainingArgs({}, only_required=False))
        plan.set_model_params(
            {"coef_": np.array([1.0, 0.0]), "intercept_": np.array([0.0])}
        )
        metrics = plan.testing_step(np.array([[1.0, 5.0]]), np.array([[3.0]]))
        assert metrics == {"MEAN_SQUARE_ERROR": 4.0}
