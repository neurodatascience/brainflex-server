#!/usr/bin/env python3

"""Run the federated learning experiments."""

import argparse
import importlib.util
import json
import math
import shutil
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from declearn.optimizer.modules import ScaffoldServerModule
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.common.optimizers.optimizer import Optimizer
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import BaseTrainingPlan
from fedbiomed.researcher.datasets import FederatedDataset
from fedbiomed.researcher.federated_workflows import Experiment, FederatedAnalytics
from fedbiomed.researcher.federated_workflows.jobs import TrainingJob

DPATH_TRAINING_PLANS = Path(__file__).resolve().parents[1] / "training_plans"
PREFIX_MODEL = "model"
PREFIX_RESULTS = "results"
PREFIX_TENSORBOARD = "tensorboard"
COMMON_TAG = "brainflex"

AGGREGATORS = ["fedavg", "fedprox", "scaffold"]

DEFAULT_N_ROUNDS = 30
DEFAULT_BATCH_SIZE = 1000  # full batch
DEFAULT_N_UPDATES = 50
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_TEST_RATIO = 0.1
DEFAULT_AGGREGATOR = "scaffold"
DEFAULT_FEDPROX_MU = 1.0
DEFAULT_ALPHA = 0.1


@dataclass(frozen=True)
class TaskConfig:
    training_plan_module: str
    training_plan_class: str
    target_column: str
    test_metric: MetricTypes
    # whether the model intercept is initialized to the global target mean
    with_target_mean: bool = False
    extra_model_args: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PathHelper:
    output_dir: Path
    n_rounds: int
    batch_size: int
    n_updates: int
    learning_rate: float
    test_ratio: float
    aggregator: str
    alpha: float

    def get_fpath_out(
        self,
        task: str,
        prefix: str,
        extension: str = ".json",
        held_out_node: str | None = None,
    ) -> Path:
        stem = "-".join(
            [
                prefix,
                self.aggregator,
                f"{self.n_rounds}rounds",
                f"{self.n_updates}updates",
                f"{self.batch_size}",
                f"{self.learning_rate}lr",
                f"{self.test_ratio}test",
                f"{self.alpha}alpha",
            ]
        )
        if held_out_node is not None:
            stem = f"{stem}-heldout_{held_out_node}"
        return self.output_dir / task / f"{stem}{extension}"


TASKS = {
    "age": TaskConfig(
        training_plan_module="tabular_regressor",
        training_plan_class="TabularRegressorTrainingPlan",
        target_column="nb:Age",
        test_metric=MetricTypes.MEAN_SQUARE_ERROR,
        with_target_mean=True,
    ),
    "sex": TaskConfig(
        training_plan_module="tabular_classifier",
        training_plan_class="TabularClassifierTrainingPlan",
        target_column="nb:Sex",
        test_metric=MetricTypes.ACCURACY,
        extra_model_args={"n_classes": 2},
    ),
}


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "output_dir", type=Path, help="Directory to save the global models to."
    )
    parser.add_argument(
        "--task",
        dest="tasks",
        action="append",
        required=True,
        choices=list(TASKS),
        help="Task to run. Can be repeated.",
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=DEFAULT_N_ROUNDS,
        help=f"Number of FL rounds. Default: {DEFAULT_N_ROUNDS}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Local batch size. Default: {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--n-updates",
        type=int,
        default=DEFAULT_N_UPDATES,
        help=f"Local updates per round. Default: {DEFAULT_N_UPDATES}.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"SGD learning rate. Default: {DEFAULT_LEARNING_RATE}.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help=f"Fraction of node data for validation. Default: {DEFAULT_TEST_RATIO}.",
    )
    parser.add_argument(
        "--aggregator",
        choices=AGGREGATORS,
        default=DEFAULT_AGGREGATOR,
        help=f"Aggregation strategy. Default: {DEFAULT_AGGREGATOR}.",
    )
    parser.add_argument(
        "--fedprox-mu",
        type=float,
        default=DEFAULT_FEDPROX_MU,
        help=f"FedProx proximal term (fedprox only). Default: {DEFAULT_FEDPROX_MU}.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"L2 regularization strength. Default: {DEFAULT_ALPHA}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results.",
    )
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save node metrics as TensorBoard logs. Default: True.",
    )
    parser.add_argument(
        "--cv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Leave-one-node-out cross-validation: train on all nodes but one and"
            " evaluate on the held-out node. Default: True."
        ),
    )
    return parser


def _filter_feature_col(col: str) -> bool:
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


def _serialize(obj: object) -> object:
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def _save_json(fpath: Path, data: dict) -> None:
    fpath.write_text(
        json.dumps(data, indent=4, allow_nan=False, default=_serialize) + "\n"
    )


def _load_training_plan_class(module_name: str, class_name: str) -> type:
    spec = importlib.util.spec_from_file_location(
        f"training_plans.{module_name}",
        DPATH_TRAINING_PLANS / f"{module_name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    # registered so that Fed-BioMed can get the source code to send to nodes
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _get_node_data(task: str) -> dict[str, dict]:
    """Get the dataset metadata of each node that has data for a task."""
    fds = Experiment(tags=[COMMON_TAG, task]).training_data()
    return fds.data() if fds is not None else {}


def _get_feature_stats(
    analytics: FederatedAnalytics, target_column: str
) -> tuple[list[str], list[float], list[float], dict[str, list[str]], float]:
    """Get the feature/excluded columns, some stats, and the target variable's mean."""
    result = analytics.fetch_stats(["mean", "std"])
    mean = result.global_stats("mean")
    std = result.global_stats("std")
    excluded_columns: dict[str, list[str]] = defaultdict(list)
    columns: list[str] = []
    for column in mean:
        if not _filter_feature_col(column):
            excluded_columns["not in feature column list"].append(column)
            continue
        if not math.isfinite(std[column]) or std[column] == 0:
            excluded_columns["zero or infinite standard deviation"].append(column)
            continue
        if column != target_column:
            columns.append(column)

    columns = sorted(columns)

    if excluded_columns:
        for reason, cols in excluded_columns.items():
            print(f"Excluding {len(cols)} features ({reason}): {cols}")

    # cast to float so that the model args are JSON-serializable
    return (
        columns,
        [float(mean[column]) for column in columns],
        [float(std[column]) for column in columns],
        excluded_columns,
        float(mean[target_column]),
    )


def _build_model_args(
    task: str,
    columns: list[str],
    mean: list[float],
    std: list[float],
    alpha: float,
    target_mean: float,
) -> dict:
    config = TASKS[task]
    model_args = {
        "n_features": len(columns),
        "feature_columns": columns,
        "feature_mean": mean,
        "feature_std": std,
        "target_column": config.target_column,
        "alpha": alpha,
        **config.extra_model_args,
    }
    if config.with_target_mean:
        model_args["target_mean"] = target_mean
    return model_args


def _build_optimizer_args(
    learning_rate: float, aggregator: str, fedprox_mu: float
) -> dict:
    return {
        "lr": learning_rate,
        "scaffold": aggregator == "scaffold",
        "fedprox_mu": fedprox_mu if aggregator == "fedprox" else None,
    }


def _get_agg_optimizer(aggregator: str) -> Optimizer | None:
    if aggregator == "scaffold":
        return Optimizer(lr=1.0, modules=[ScaffoldServerModule()])
    return None


def _move_tensorboard_logs(exp: Experiment, dpath_tensorboard: Path) -> None:
    """Move the per-node logs out of the Fed-BioMed experiment directory."""
    # flush the event files before moving them
    exp.monitor().close_writer()
    dpath_tensorboard.mkdir(parents=True, exist_ok=True)
    for dpath_node_src in Path(exp.tensorboard_results_path).glob("*"):
        dpath_node_dest = dpath_tensorboard / dpath_node_src.name
        if dpath_node_dest.exists():
            shutil.rmtree(dpath_node_dest)
        shutil.move(dpath_node_src, dpath_node_dest)


def _evaluate_held_out(exp: Experiment, node_data_held_out: dict[str, dict]) -> None:
    """Test the current global model on all the data of the held-out node.

    Same as the validation-only round of ``Experiment.run_once(test_after=True)``,
    but on a node that is not part of the experiment.
    """
    training_args = TrainingArgs(
        {
            **exp.training_args(),
            "test_ratio": 1.0,
            "test_on_global_updates": True,
            "test_on_local_updates": False,
            "shuffle_testing_dataset": False,
        },
        only_required=False,
    )
    TrainingJob(
        researcher_id=exp.researcher_id,
        requests=exp.requests,
        nodes=list(node_data_held_out),
        keep_files_dir=exp.experimentation_path(),
        experiment_id=exp.id,
        round_=exp.round_current(),
        training_plan=exp.training_plan(),
        training_args=training_args,
        model_args=exp.model_args(),
        data=FederatedDataset(node_data_held_out),
        nodes_state_ids={},
        aggregator_args={},
        do_training=False,
    ).execute()


def _run_task(
    task: str,
    n_rounds: int,
    batch_size: int,
    n_updates: int,
    learning_rate: float,
    test_ratio: float,
    aggregator: str,
    fedprox_mu: float,
    alpha: float,
    settings: dict,
    fpath_results: Path,
    fpath_model: Path,
    dpath_tensorboard: Path | None = None,
    training_nodes: list[str] | None = None,
    node_data_held_out: dict[str, dict] | None = None,
) -> BaseTrainingPlan:
    """Train a model for a task and save it in a task-specific output directory.

    If ``node_data_held_out`` is given, only ``training_nodes`` are used for training
    and the held-out node is used for testing.
    """
    config = TASKS[task]
    is_cv = node_data_held_out is not None
    exp = Experiment(
        tags=[COMMON_TAG, task],
        nodes=training_nodes,
        training_args={
            "loader_args": {"batch_size": batch_size},
            "num_updates": n_updates,
            "test_ratio": test_ratio,
            "test_metric": config.test_metric,
            "test_on_global_updates": True,
            "test_on_local_updates": True,
            "optimizer_args": _build_optimizer_args(
                learning_rate, aggregator, fedprox_mu
            ),
            # resample the local train/validation split at each round
            "shuffle_testing_dataset": is_cv,
        },
        agg_optimizer=_get_agg_optimizer(aggregator),
        round_limit=n_rounds,
        tensorboard=dpath_tensorboard is not None,
    )

    if not is_cv:
        warnings.warn(
            "Feature z-scoring statistics are computed on all node data, including"
            " the test data (data leakage). Use --cv to avoid this",
            stacklevel=2,
        )
        if config.with_target_mean:
            warnings.warn(
                "The intercept is initialized with the target mean of all node data,"
                " including the test data (data leakage). Use --cv to avoid this",
                stacklevel=2,
            )

    # the training plan needs the feature stats to be instantiated
    columns, mean, std, excluded_columns, target_mean = _get_feature_stats(
        exp.analytics, config.target_column
    )
    print(f"[{task}] Using {len(columns)} features")
    model_args = _build_model_args(task, columns, mean, std, alpha, target_mean)
    exp.set_model_args(model_args)
    exp.set_training_plan_class(
        _load_training_plan_class(
            config.training_plan_module, config.training_plan_class
        )
    )

    # saved before training so that they are available even if training fails
    json_data = {
        "settings": {
            "runtime_args": settings,
            "training_nodes": training_nodes,
            "held_out_node": node_data_held_out,
            "excluded_columns": excluded_columns,
            "model_args": exp.model_args(),
            "training_args": exp.training_args(),
        }
    }
    fpath_results.parent.mkdir(parents=True, exist_ok=True)
    _save_json(fpath_results, json_data)

    for round_ in range(1, n_rounds + 1):
        print(f"[{task}] Round {round_}/{n_rounds}")
        if is_cv:
            _evaluate_held_out(exp, node_data_held_out)
        exp.run_once()
        json_data["results"] = {
            # node training losses and test metrics (no public getter)
            "node_metrics": exp.monitor()._metric_store,
            "training_replies": exp.save_training_replies(),
        }
        _save_json(fpath_results, json_data)
    if is_cv:
        # final global model
        _evaluate_held_out(exp, node_data_held_out)
        json_data["results"]["node_metrics"] = exp.monitor()._metric_store
        _save_json(fpath_results, json_data)
    print(f"[{task}] Saved settings and results to {fpath_results}")

    if dpath_tensorboard is not None:
        _move_tensorboard_logs(exp, dpath_tensorboard)
        print(f"[{task}] Saved TensorBoard logs to {dpath_tensorboard}")

    fpath_model.parent.mkdir(parents=True, exist_ok=True)
    exp.training_plan().export_model(str(fpath_model))
    print(f"[{task}] Saved global model to {fpath_model}")
    return exp.training_plan()


def run_experiments(
    output_dir: Path,
    tasks: list[str],
    n_rounds: int = DEFAULT_N_ROUNDS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_updates: int = DEFAULT_N_UPDATES,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    test_ratio: float = DEFAULT_TEST_RATIO,
    aggregator: str = DEFAULT_AGGREGATOR,
    fedprox_mu: float = DEFAULT_FEDPROX_MU,
    alpha: float = DEFAULT_ALPHA,
    overwrite: bool = False,
    tensorboard: bool = True,
    cv: bool = True,
) -> None:
    """Train and save a global model for each task."""
    output_dir = output_dir.resolve()
    settings = locals().copy()

    # (task, held-out node) pairs, with None for no held-out node
    runs: list[tuple[str, str | None]] = []
    node_data: dict[str, dict[str, dict]] = {}
    for task in tasks:
        if not cv:
            runs.append((task, None))
            continue
        node_data[task] = _get_node_data(task)
        if len(node_data[task]) < 2:
            raise ValueError(
                f"CV needs at least 2 nodes, but {task} has {len(node_data[task])}"
            )
        runs.extend((task, node) for node in sorted(node_data[task]))

    path_helper = PathHelper(
        output_dir=output_dir,
        n_rounds=n_rounds,
        batch_size=batch_size,
        n_updates=n_updates,
        learning_rate=learning_rate,
        test_ratio=test_ratio,
        aggregator=aggregator,
        alpha=alpha,
    )

    # error out if one of the result files already exists
    fpaths_results_existing = []
    for task, held_out_node in runs:
        fpath = path_helper.get_fpath_out(
            task=task,
            prefix=PREFIX_RESULTS,
            extension=".json",
            held_out_node=held_out_node,
        )
        if fpath.exists() and "results" in json.loads(fpath.read_text()):
            fpaths_results_existing.append(fpath)

    if fpaths_results_existing and not overwrite:
        raise FileExistsError(
            f"Results already exist: {[str(fpath) for fpath in fpaths_results_existing]}."
            " Use --overwrite to overwrite them"
        )

    for task, held_out_node in runs:
        if held_out_node is None:
            # use all nodes for training
            training_nodes = None
            node_data_held_out = None
        else:
            print(f"[{task}] Holding out node {held_out_node}")
            training_nodes = sorted(set(node_data[task]) - {held_out_node})
            node_data_held_out = {held_out_node: node_data[task][held_out_node]}

        _run_task(
            task=task,
            n_rounds=n_rounds,
            batch_size=batch_size,
            n_updates=n_updates,
            learning_rate=learning_rate,
            test_ratio=test_ratio,
            aggregator=aggregator,
            fedprox_mu=fedprox_mu,
            alpha=alpha,
            settings=settings,
            fpath_results=path_helper.get_fpath_out(
                task=task,
                prefix=PREFIX_RESULTS,
                extension=".json",
                held_out_node=held_out_node,
            ),
            fpath_model=path_helper.get_fpath_out(
                task=task,
                prefix=PREFIX_MODEL,
                extension=".pkl",
                held_out_node=held_out_node,
            ),
            dpath_tensorboard=(
                path_helper.get_fpath_out(
                    task=task,
                    prefix=PREFIX_TENSORBOARD,
                    extension="",
                    held_out_node=held_out_node,
                )
                if tensorboard
                else None
            ),
            training_nodes=training_nodes,
            node_data_held_out=node_data_held_out,
        )


def main(argv: list[str] | None = None) -> None:
    args = _get_parser().parse_args(argv)
    try:
        run_experiments(
            args.output_dir,
            args.tasks,
            n_rounds=args.n_rounds,
            batch_size=args.batch_size,
            n_updates=args.n_updates,
            learning_rate=args.learning_rate,
            test_ratio=args.test_ratio,
            aggregator=args.aggregator,
            fedprox_mu=args.fedprox_mu,
            alpha=args.alpha,
            overwrite=args.overwrite,
            tensorboard=args.tensorboard,
            cv=args.cv,
        )
    except FileExistsError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
