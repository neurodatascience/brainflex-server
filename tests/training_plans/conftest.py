import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from fedbiomed.common.dataset import Dataset
from fedbiomed.common.optimizers.generic_optimizers import DeclearnOptimizer
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import BaseTrainingPlan

DPATH_TRAINING_PLANS = Path(__file__).parents[2] / "training_plans"


@pytest.fixture
def load_training_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str, str], type[BaseTrainingPlan]]:
    """Load a training plan class from the training_plans directory."""

    def _load(module_name: str, class_name: str) -> type[BaseTrainingPlan]:
        spec = importlib.util.spec_from_file_location(
            f"training_plans.{module_name}",
            DPATH_TRAINING_PLANS / f"{module_name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return getattr(module, class_name)

    return _load


@pytest.fixture
def fpath_csv(tmp_path: Path) -> Path:
    """Node CSV dataset, as written by the nipoppy2fedbiomed pipeline."""
    fpath = tmp_path / "ds000001.csv"
    pd.DataFrame(
        {
            "nb:Age": [20.0, 30.0, 40.0],
            "nb:Sex": [0, 1, 1],
            "Left-Hippocampus": [4100.0, 4000.0, 3900.0],
            "lh_insula": [2.0, 2.5, 3.0],
        }
    ).to_csv(fpath, index=False)
    return fpath


@pytest.fixture
def model_args() -> dict:
    return {
        "n_features": 2,
        "feature_columns": ["Left-Hippocampus", "lh_insula"],
        "feature_mean": [4000.0, 2.5],
        "feature_std": [100.0, 0.5],
    }


@pytest.fixture
def optimizer_config() -> Callable[[BaseTrainingPlan, dict, dict], dict | None]:
    """Set up a training plan and get its declearn optimizer config, if any."""
    return _optimizer_config


@pytest.fixture
def load_dataset() -> Callable[[BaseTrainingPlan, dict, Path], Dataset]:
    """Set up a training plan and load its dataset, like a Fed-BioMed node does."""
    return _load_dataset


@pytest.fixture
def from_source() -> Callable[[BaseTrainingPlan], type[BaseTrainingPlan]]:
    """Rebuild a training plan class from the source code sent to nodes."""
    return _from_source


def _optimizer_config(
    plan: BaseTrainingPlan, model_args: dict, optimizer_args: dict
) -> dict | None:
    plan.post_init(
        dict(model_args),
        TrainingArgs({"optimizer_args": optimizer_args}, only_required=False),
    )
    if not isinstance(plan.optimizer(), DeclearnOptimizer):
        return None
    return plan.optimizer().optimizer.get_state()["config"]


def _load_dataset(plan: BaseTrainingPlan, model_args: dict, fpath: Path) -> Dataset:
    plan.post_init(dict(model_args), TrainingArgs({}, only_required=False))
    data_manager = plan.training_data()
    data_manager.load(tp_type=plan.type())
    data_manager.load_dataset(root=str(fpath))
    return data_manager._dataset


def _from_source(plan: BaseTrainingPlan) -> type[BaseTrainingPlan]:
    namespace = {}
    exec(plan.source(), namespace)  # noqa: S102
    return namespace[type(plan).__name__]
