"""Configuration of the experiment tasks."""

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

from fedbiomed.common.metrics import MetricTypes

DPATH_TRAINING_PLANS = Path(__file__).resolve().parents[1] / "training_plans"

COMMON_TAG = "brainflex"


@dataclass(frozen=True)
class TaskConfig:
    training_plan_module: str
    training_plan_class: str
    target_column: str
    test_metric: MetricTypes
    # whether the model intercept is initialized to the global target mean
    with_target_mean: bool = False
    extra_model_args: dict = field(default_factory=dict)
    # model parameters are clipped to [-x, x] before being encrypted for secure
    # aggregation
    secagg_clipping_range: int | None = None


TASKS = {
    "age": TaskConfig(
        training_plan_module="tabular_regressor",
        training_plan_class="TabularRegressorTrainingPlan",
        target_column="nb:Age",
        test_metric=MetricTypes.MEAN_SQUARE_ERROR,
        with_target_mean=True,
        secagg_clipping_range=100,
    ),
    "sex": TaskConfig(
        training_plan_module="tabular_classifier",
        training_plan_class="TabularClassifierTrainingPlan",
        target_column="nb:Sex",
        test_metric=MetricTypes.ACCURACY,
        extra_model_args={"n_classes": 2},
    ),
}


def filter_feature_col(col: str) -> bool:
    if col.startswith("nb:"):
        return True
    elif col.endswith("_thickness"):
        # return False
        return not (
            col
            in [
                # exclude these
                "lh_MeanThickness_thickness",
                "rh_MeanThickness_thickness",
                "rh_temporalpole_thickness",
            ]
        )
    else:
        return col in [
            # include these
            "EstimatedTotalIntraCranialVol",
            "Left-Lateral-Ventricle",
            "Left-Thalamus",
            "Left-Caudate",
            "Left-Putamen",
            "Left-Pallidum",
            "Left-Hippocampus",
            "Left-Amygdala",
            "Left-Accumbens-area",
            "Right-Lateral-Ventricle",
            "Right-Thalamus",
            "Right-Caudate",
            "Right-Putamen",
            "Right-Pallidum",
            "Right-Hippocampus",
            "Right-Amygdala",
            "Right-Accumbens-area",
        ]


def load_training_plan_class(module_name: str, class_name: str) -> type:
    spec = importlib.util.spec_from_file_location(
        f"training_plans.{module_name}",
        DPATH_TRAINING_PLANS / f"{module_name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    # registered so that Fed-BioMed can get the source code to send to nodes
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)
