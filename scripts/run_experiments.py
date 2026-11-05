#!/usr/bin/env python3

"""Run the federated learning experiments."""

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from collections.abc import Generator
from datetime import date
from pathlib import Path

from _io_utils import (
    PREFIX_LOGS,
    PREFIX_MODEL,
    PREFIX_RESULTS,
    PREFIX_TENSORBOARD,
    PathHelper,
    save_json,
)
from _log_utils import log_to_file, logger, print_logger_messages_at_exit
from _tasks import COMMON_TAG, TASKS, filter_feature_col
from declearn.optimizer.modules import ScaffoldServerModule
from fedbiomed.common.optimizers.optimizer import Optimizer
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.researcher.datasets import FederatedDataset
from fedbiomed.researcher.federated_workflows import Experiment, FederatedAnalytics
from fedbiomed.researcher.federated_workflows.jobs import TrainingJob
from fedbiomed.researcher.secagg import SecureAggregation

AGGREGATORS = ["fedavg", "fedprox", "scaffold"]

DEFAULT_TASKS = ["age", "sex"]
DEFAULT_N_ROUNDS = 30
DEFAULT_BATCH_SIZE = 1000  # full batch
DEFAULT_N_UPDATES = 50
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_TEST_RATIO = 0.1
DEFAULT_AGGREGATOR = "scaffold"
DEFAULT_FEDPROX_MU = 1.0
DEFAULT_ALPHA = 0.1
DEFAULT_TENSORBOARD = True
DEFAULT_SECAGG = True
DEFAULT_CV = True
DEFAULT_RANDOM_SEED = 3791
DEFAULT_DATE_TAG = date.today().strftime("%Y_%m_%d")


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "output_dir", type=Path, help="Directory to save the global models to."
    )
    parser.add_argument(
        "--date-tag",
        default=DEFAULT_DATE_TAG,
        help=(
            "Name of the subfolder of the output directory to save results to."
            f" Default: {DEFAULT_DATE_TAG}."
        ),
    )
    parser.add_argument(
        "--task",
        dest="tasks",
        action="append",
        choices=list(TASKS),
        help=f"Task to run. Can be repeated. Default: {DEFAULT_TASKS}",
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
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TENSORBOARD,
        help=f"Save node metrics as TensorBoard logs. Default: {DEFAULT_TENSORBOARD}.",
    )
    parser.add_argument(
        "--secagg",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SECAGG,
        help=(
            "Secure aggregation of the model parameters and the feature statistics."
            f" Needs at least 2 training nodes. Default: {DEFAULT_SECAGG}."
        ),
    )
    parser.add_argument(
        "--cv",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CV,
        help=(
            "Leave-one-node-out cross-validation: train on all nodes but one and"
            f" evaluate on the held-out node. Default: {DEFAULT_CV}."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=(
            "Random seed for the node data splits, batch shuffling and SGD."
            f" Incremented at each round. Default: {DEFAULT_RANDOM_SEED}."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results.",
    )
    return parser


def _get_node_data(task: str) -> dict[str, dict]:
    """Get the dataset metadata of each node that has data for a task."""
    fds = Experiment(tags=[COMMON_TAG, task]).training_data()
    return fds.data() if fds is not None else {}


def _get_column_stats(
    analytics: FederatedAnalytics, node_data: dict[str, dict], target_column: str
) -> tuple[list[str], list[float], list[float], dict[str, list[str]], float]:
    """Get the feature/excluded columns, some stats, and the target variable's mean."""
    if not node_data:
        raise ValueError("No node has a dataset for this task")
    excluded_columns: dict[str, list[str]] = defaultdict(list)
    column_sets = [set(dataset["dtypes"]) for dataset in node_data.values()]
    common_columns = set.intersection(*column_sets)
    if target_column not in common_columns:
        raise ValueError(
            f"Target column {target_column} is not in the datasets of all nodes"
        )
    if missing_columns := sorted(set.union(*column_sets) - common_columns):
        excluded_columns["not in the datasets of all nodes"] = missing_columns
    stats_columns: list[str] = []
    for column in sorted(common_columns):
        if column == target_column or filter_feature_col(column):
            stats_columns.append(column)
        else:
            excluded_columns["not in feature column list"].append(column)

    # with secure aggregation, all nodes must compute the stats of the same columns
    result = analytics.fetch_stats(["mean", "std"], dataset_schema=stats_columns)
    mean = result.global_stats("mean")
    std = result.global_stats("std")
    columns: list[str] = []
    for column in stats_columns:
        if not math.isfinite(std[column]) or std[column] == 0:
            excluded_columns["zero or infinite standard deviation"].append(column)
            continue
        if column != target_column:
            columns.append(column)

    columns = sorted(columns)

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
        # all the metrics supported by the training plan
        "metrics": list(config.training_plan_class.METRICS),
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
    """Move the Tensorboard logs out of the Fed-BioMed experiment directory."""
    # flush the event files before moving them
    exp.monitor().close_writer()
    dpath_tensorboard.mkdir(parents=True, exist_ok=True)
    for dpath_node_src in Path(exp.tensorboard_results_path).glob("*"):
        dpath_node_dest = dpath_tensorboard / dpath_node_src.name
        if dpath_node_dest.exists():
            shutil.rmtree(dpath_node_dest)
        shutil.move(dpath_node_src, dpath_node_dest)


def _setup_held_out_secagg(exp: Experiment, nodes: list[str]) -> dict:
    """Set up secure aggregation for the requests to the held-out node."""
    secagg = SecureAggregation()
    secagg.setup(
        parties=nodes,
        experiment_id=exp.id,
        researcher_id=exp.researcher_id,
        insecure_validation=False,
    )
    return secagg.train_arguments()


def _evaluate_held_out(
    exp: Experiment,
    node_data_held_out: dict[str, dict],
    secagg_arguments: dict | None = None,
) -> None:
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
    replies, _ = TrainingJob(
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
        secagg_arguments=secagg_arguments,
        do_training=False,
    ).execute()
    # raise explicit errors, otherwise failed requests are silent
    for node in node_data_held_out:
        if node not in replies:
            raise RuntimeError(f"Held-out node {node} did not reply")
        if not replies[node]["success"]:
            raise RuntimeError(
                f"Held-out node {node} failed to evaluate the model:"
                f" {replies[node]['msg']}"
            )


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
    fpath_model: Path,
    dpath_tensorboard: Path | None = None,
    training_nodes: list[str] | None = None,
    node_data_held_out: dict[str, dict] | None = None,
    secagg: bool = True,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Generator[tuple[str, dict], None, None]:
    """Train a model for a task and save it in a task-specific output directory.

    If ``node_data_held_out`` is given, only ``training_nodes`` are used for training
    and the held-out node is used for testing.

    Yields the fold key (the training nodes joined by ``__``) and the fold settings
    and results. Saves the model and TensorBoard logs after the last yield.
    """
    config = TASKS[task]
    is_cv = node_data_held_out is not None
    held_out_node = next(iter(node_data_held_out)) if is_cv else None
    section_name = task.upper()
    if is_cv:
        section_name += f": held_out_node={held_out_node}"
    logger.info(f"===== {section_name} =====")
    exp = Experiment(
        tags=[COMMON_TAG, task],
        nodes=training_nodes,
        training_args={
            "loader_args": {"batch_size": batch_size},
            "num_updates": n_updates,
            "test_ratio": test_ratio,
            "test_on_global_updates": True,
            "test_on_local_updates": True,
            "optimizer_args": _build_optimizer_args(
                learning_rate, aggregator, fedprox_mu
            ),
            # resample the local train/validation split at each round
            "shuffle_testing_dataset": is_cv,
            "random_seed": random_seed,
        },
        agg_optimizer=_get_agg_optimizer(aggregator),
        round_limit=n_rounds,
        tensorboard=dpath_tensorboard is not None,
        secagg=SecureAggregation(
            active=secagg, clipping_range=config.secagg_clipping_range
        ),
    )
    if training_nodes is not None:
        # redo the search to drop the held-out node
        # otherwise, it would be included in the federated analytics
        exp.set_training_data(None, from_tags=True)
    fold_key = "__".join(sorted(exp.training_data().data()))

    if not is_cv:
        logger.warning(
            "Feature z-scoring statistics are computed on all node data,"
            " including the test data (data leakage). Use --cv to avoid this"
        )
        if config.with_target_mean:
            logger.warning(
                "The intercept is initialized with the target mean of all"
                " node data, including the test data (data leakage). Use --cv to"
                " avoid this"
            )

    # the training plan needs the feature stats to be instantiated
    columns, mean, std, excluded_columns, target_mean = _get_column_stats(
        exp.analytics, exp.training_data().data(), config.target_column
    )
    for reason, cols in excluded_columns.items():
        logger.info(f"Excluding {len(cols)} features ({reason}): {cols}")
    logger.info(f"Using {len(columns)} features")
    model_args = _build_model_args(task, columns, mean, std, alpha, target_mean)
    exp.set_model_args(model_args)
    exp.set_training_plan_class(config.training_plan_class)

    # saved before training so that they are available even if training fails
    fold = {
        "settings": {
            "training_nodes": training_nodes,
            "held_out_node": node_data_held_out,
            "excluded_columns": excluded_columns,
            "model_args": exp.model_args(),
            "training_args": exp.training_args(),
        }
    }
    yield fold_key, fold

    held_out_secagg_arguments = None
    if is_cv and secagg:
        held_out_secagg_arguments = _setup_held_out_secagg(
            exp, [*training_nodes, *node_data_held_out]
        )

    for round_ in range(1, n_rounds + 1):
        logger.info(f"Round {round_}/{n_rounds}")
        # nodes reseed at the start of every round, so offset the seed to get a
        # different (but deterministic) data split and batch order at each round
        exp.set_training_args(
            {**exp.training_args(), "random_seed": random_seed + round_}
        )
        if is_cv:
            _evaluate_held_out(exp, node_data_held_out, held_out_secagg_arguments)
        exp.run_once()
        fold["results"] = {
            # node training losses and test metrics (no public getter)
            "node_metrics": exp.monitor()._metric_store,
            "training_replies": exp.save_training_replies(),
        }
        yield fold_key, fold
    if is_cv:
        # final global model
        _evaluate_held_out(exp, node_data_held_out, held_out_secagg_arguments)
        fold["results"]["node_metrics"] = exp.monitor()._metric_store
        yield fold_key, fold

    if dpath_tensorboard is not None:
        _move_tensorboard_logs(exp, dpath_tensorboard)
        logger.info(f"Saved TensorBoard logs to {dpath_tensorboard}")

    fpath_model.parent.mkdir(parents=True, exist_ok=True)
    exp.training_plan().export_model(str(fpath_model))
    logger.info(f"Saved global model to {fpath_model}")


def run_experiments(
    output_dir: Path,
    tasks: list[str],
    date_tag: str = DEFAULT_DATE_TAG,
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
    secagg: bool = True,
    cv: bool = True,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> None:
    """Train and save a global model for each task."""
    output_dir = output_dir.resolve()
    settings = locals().copy()

    # keys: task
    # values: held-out node (None if not doing CV)
    runs: dict[str, list[str | None]] = {}
    node_data: dict[str, dict[str, dict]] = {}
    for task in tasks:
        if not cv:
            runs[task] = [None]
            continue
        node_data[task] = _get_node_data(task)
        # secure aggregation needs at least 2 training nodes
        min_n_nodes = 3 if secagg else 2
        if len(node_data[task]) < min_n_nodes:
            raise ValueError(
                f"CV needs at least {min_n_nodes} nodes"
                f"{' with secure aggregation' if secagg else ''},"
                f" but {task} has {len(node_data[task])}"
            )
        runs[task] = sorted(node_data[task])

    path_helper = PathHelper(
        output_dir=output_dir,
        date_tag=date_tag,
        n_rounds=n_rounds,
        batch_size=batch_size,
        n_updates=n_updates,
        learning_rate=learning_rate,
        test_ratio=test_ratio,
        aggregator=aggregator,
        fedprox_mu=fedprox_mu,
        alpha=alpha,
        secagg=secagg,
        cv=cv,
        random_seed=random_seed,
    )

    # error out if one of the result files already exists
    fpaths_results_existing = []
    for task in runs:
        fpath = path_helper.get_fpath_out(
            task=task, prefix=PREFIX_RESULTS, extension=".json"
        )
        if fpath.exists() and any(
            "results" in fold
            for fold in json.loads(fpath.read_text()).get("folds", {}).values()
        ):
            fpaths_results_existing.append(fpath)

    if fpaths_results_existing and not overwrite:
        raise FileExistsError(
            f"Results already exist: {[str(fpath) for fpath in fpaths_results_existing]}."
            " Use --overwrite to overwrite them"
        )

    with print_logger_messages_at_exit():
        for task, held_out_nodes in runs.items():
            fpath_results = path_helper.get_fpath_out(
                task=task, prefix=PREFIX_RESULTS, extension=".json"
            )
            fpath_log = path_helper.get_fpath_out(
                task=task, prefix=PREFIX_LOGS, extension=".log"
            )
            json_data = {"settings": {"runtime_args": settings}, "folds": {}}
            fpath_results.parent.mkdir(parents=True, exist_ok=True)
            with log_to_file(fpath_log):
                for held_out_node in held_out_nodes:
                    if held_out_node is None:
                        # use all nodes for training
                        training_nodes = None
                        node_data_held_out = None
                    else:
                        training_nodes = sorted(set(node_data[task]) - {held_out_node})
                        node_data_held_out = {
                            held_out_node: node_data[task][held_out_node]
                        }

                    for fold_key, fold in _run_task(
                        task=task,
                        n_rounds=n_rounds,
                        batch_size=batch_size,
                        n_updates=n_updates,
                        learning_rate=learning_rate,
                        test_ratio=test_ratio,
                        aggregator=aggregator,
                        fedprox_mu=fedprox_mu,
                        alpha=alpha,
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
                        secagg=secagg,
                        random_seed=random_seed,
                    ):
                        json_data["folds"][fold_key] = fold
                        save_json(fpath_results, json_data)
                    logger.info(f"Saved settings and results to {fpath_results}")
            logger.info(f"Saved log to {fpath_log}")


def main(argv: list[str] | None = None) -> None:
    args = _get_parser().parse_args(argv)
    try:
        run_experiments(
            args.output_dir,
            args.tasks if args.tasks is not None else DEFAULT_TASKS,
            date_tag=args.date_tag,
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
            secagg=args.secagg,
            cv=args.cv,
            random_seed=args.random_seed,
        )
    except FileExistsError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
