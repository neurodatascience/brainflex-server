from collections.abc import Callable
from types import ModuleType

import pytest
from fedbiomed.common.training_plans import BaseTrainingPlan


@pytest.fixture
def module(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("_tasks")


class TestLoadTrainingPlanClass:
    @pytest.mark.parametrize("task", ["age", "sex"])
    def test_source(self, module: ModuleType, task: str):
        plan_cls = module.TASKS[task].training_plan_class
        assert issubclass(plan_cls, BaseTrainingPlan)
        assert f"class {plan_cls.__name__}" in plan_cls().source()


class TestFilterFeatureCol:
    @pytest.mark.parametrize(
        "col",
        [
            "nb:Age",
            "nb:Sex",
            "EstimatedTotalIntraCranialVol",
            "Left-Hippocampus",
            "Right-Accumbens-area",
        ],
    )
    def test_included(self, module: ModuleType, col: str):
        assert module.filter_feature_col(col)

    @pytest.mark.parametrize(
        "col",
        [
            "Left-WM-hypointensities",
            "non-WM-hypointensities",
            "CSF",
            "5th-Ventricle",
            "lhSurfaceHoles",
            "lh_insula",
        ],
    )
    def test_excluded(self, module: ModuleType, col: str):
        assert not module.filter_feature_col(col)
