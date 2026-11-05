import json
import logging
import sys
from collections.abc import Callable, Generator
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fedbiomed.common.training_args import TrainingArgs
from pytest_mock import MockerFixture

GLOBAL_STATS = {
    "mean": {
        "nb:Age": np.float64(30.0),
        "nb:Sex": np.float64(0.5),
        "Left-Thalamus": np.float64(7000.0),
        "Left-Hippocampus": np.float64(4000.0),
    },
    "std": {
        "nb:Age": np.float64(10.0),
        "nb:Sex": np.float64(0.5),
        "Left-Thalamus": np.float64(50.0),
        "Left-Hippocampus": np.float64(100.0),
    },
}
NODE_METRICS = {
    "NODE_1": {
        "testing_global_updates": {
            "MEAN_SQUARE_ERROR": {1: {"iterations": [1], "values": [12.5]}}
        }
    }
}
TRAINING_REPLIES = {0: {"NODE_1": {"success": True, "sample_size": 80}}}
NODE_DATA = {
    node: {
        "dataset_id": f"dataset_{node}",
        "name": f"name_{node}",
        "dtypes": dict.fromkeys(GLOBAL_STATS["mean"], "Float64"),
    }
    for node in ["NODE_1", "NODE_2", "NODE_3"]
}


@pytest.fixture
def module(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("run_experiments")


@pytest.fixture
def analytics() -> MagicMock:
    analytics = MagicMock()
    analytics.fetch_stats.return_value.global_stats.side_effect = GLOBAL_STATS.get
    return analytics


@pytest.fixture
def dpath_fbm_tensorboard(tmp_path: Path) -> Path:
    """Fed-BioMed experiment TensorBoard directory with one log dir per node."""
    dpath = tmp_path / "fbm-researcher" / "tensorboard"
    for node in ["NODE_1", "NODE_2"]:
        (dpath / node).mkdir(parents=True)
        (dpath / node / "events.out.tfevents").touch()
    return dpath


@pytest.fixture
def mock_experiment(
    module: ModuleType,
    mocker: MockerFixture,
    analytics: MagicMock,
    dpath_fbm_tensorboard: Path,
) -> MagicMock:
    mock_experiment = mocker.patch.object(module, "Experiment")
    exp = mock_experiment.return_value
    exp.analytics = analytics
    exp.training_data.return_value.data.return_value = NODE_DATA
    # mimic the getters of a real Experiment
    exp.model_args.side_effect = lambda: exp.set_model_args.call_args.args[0]
    exp.training_args.side_effect = lambda: TrainingArgs(
        mock_experiment.call_args.kwargs["training_args"]
    ).dict()
    exp.monitor.return_value._metric_store = NODE_METRICS
    exp.save_training_replies.return_value = TRAINING_REPLIES
    exp.tensorboard_results_path = str(dpath_fbm_tensorboard)
    return mock_experiment


def _fake_run_task(
    training_nodes: list[str] | None = None, **kwargs
) -> Generator[tuple[str, dict], None, None]:
    """Yield a fold like _run_task, without training."""
    fold_key = "__".join(training_nodes or NODE_DATA)
    fold = {"settings": {"training_nodes": training_nodes}}
    yield fold_key, fold
    fold["results"] = {"node_metrics": {}}
    yield fold_key, fold


@pytest.fixture
def mock_run_task(module: ModuleType, mocker: MockerFixture) -> MagicMock:
    return mocker.patch.object(module, "_run_task", side_effect=_fake_run_task)


@pytest.fixture
def mock_evaluate_held_out(module: ModuleType, mocker: MockerFixture) -> MagicMock:
    return mocker.patch.object(module, "_evaluate_held_out")


@pytest.fixture
def mock_setup_held_out_secagg(module: ModuleType, mocker: MockerFixture) -> MagicMock:
    return mocker.patch.object(
        module, "_setup_held_out_secagg", return_value={"secagg_scheme": 0}
    )


@pytest.fixture
def mock_get_node_data(module: ModuleType, mocker: MockerFixture) -> MagicMock:
    return mocker.patch.object(module, "_get_node_data", return_value=NODE_DATA)


@pytest.fixture
def fpath_csv(tmp_path: Path) -> Path:
    fpath = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "nb:Age": [20.0, 30.0, 40.0],
            "nb:Sex": [0, 1, 1],
            "Left-Hippocampus": [4100.0, 4000.0, 3900.0],
            "Left-Thalamus": [6950.0, 7000.0, 7050.0],
        }
    ).to_csv(fpath, index=False)
    return fpath


class TestGetNodeData:
    def test_get_node_data(self, module: ModuleType, mock_experiment: MagicMock):
        exp = mock_experiment.return_value
        exp.training_data.return_value.data.return_value = NODE_DATA
        assert module._get_node_data("age") == NODE_DATA
        assert mock_experiment.call_args.kwargs["tags"] == ["brainflex", "age"]

    def test_no_nodes(self, module: ModuleType, mock_experiment: MagicMock):
        mock_experiment.return_value.training_data.return_value = None
        assert module._get_node_data("age") == {}


class TestEvaluateHeldOut:
    def test_evaluate_held_out(
        self, module: ModuleType, mock_experiment: MagicMock, mocker: MockerFixture
    ):
        mock_training_job = mocker.patch.object(module, "TrainingJob", autospec=True)
        mock_training_job.return_value.execute.return_value = (
            {"NODE_3": {"success": True, "msg": ""}},
            {},
        )
        mock_experiment(
            training_args={"test_ratio": 0.1, "test_on_local_updates": True}
        )
        exp = mock_experiment.return_value
        exp.round_current.return_value = 3
        exp.model_args.side_effect = None
        exp.model_args.return_value = {"n_features": 2}
        node_data_held_out = {"NODE_3": NODE_DATA["NODE_3"]}

        module._evaluate_held_out(exp, node_data_held_out)

        kwargs = mock_training_job.call_args.kwargs
        assert kwargs["nodes"] == ["NODE_3"]
        assert kwargs["do_training"] is False
        assert kwargs["round_"] == 3
        assert kwargs["experiment_id"] == exp.id
        assert kwargs["model_args"] == {"n_features": 2}
        assert kwargs["data"].data() == node_data_held_out
        training_args = kwargs["training_args"]
        assert training_args["test_ratio"] == 1.0
        assert training_args["test_on_global_updates"] is True
        assert training_args["test_on_local_updates"] is False
        assert kwargs["secagg_arguments"] is None
        mock_training_job.return_value.execute.assert_called_once_with()

    def test_secagg_arguments(
        self, module: ModuleType, mock_experiment: MagicMock, mocker: MockerFixture
    ):
        mock_training_job = mocker.patch.object(module, "TrainingJob", autospec=True)
        mock_training_job.return_value.execute.return_value = (
            {"NODE_3": {"success": True, "msg": ""}},
            {},
        )
        mock_experiment(training_args={})
        exp = mock_experiment.return_value
        exp.model_args.side_effect = None
        secagg_arguments = {"secagg_scheme": 0}
        module._evaluate_held_out(
            exp, {"NODE_3": NODE_DATA["NODE_3"]}, secagg_arguments
        )
        kwargs = mock_training_job.call_args.kwargs
        assert kwargs["secagg_arguments"] == secagg_arguments

    @pytest.mark.parametrize(
        "replies,match",
        [
            ({}, "did not reply"),
            ({"NODE_3": {"success": False, "msg": "error"}}, "failed.*: error"),
        ],
    )
    def test_failed(
        self,
        module: ModuleType,
        mock_experiment: MagicMock,
        mocker: MockerFixture,
        replies: dict,
        match: str,
    ):
        mock_training_job = mocker.patch.object(module, "TrainingJob", autospec=True)
        mock_training_job.return_value.execute.return_value = (replies, {})
        mock_experiment(training_args={})
        exp = mock_experiment.return_value
        exp.model_args.side_effect = None
        with pytest.raises(RuntimeError, match=match):
            module._evaluate_held_out(exp, {"NODE_3": NODE_DATA["NODE_3"]})


class TestSetupHeldOutSecagg:
    def test_setup_held_out_secagg(
        self, module: ModuleType, mock_experiment: MagicMock, mocker: MockerFixture
    ):
        mock_secagg = mocker.patch.object(module, "SecureAggregation")
        exp = mock_experiment.return_value
        nodes = ["NODE_1", "NODE_2", "NODE_3"]

        secagg_arguments = module._setup_held_out_secagg(exp, nodes)

        secagg = mock_secagg.return_value
        kwargs = secagg.setup.call_args.kwargs
        assert kwargs["parties"] == nodes
        assert kwargs["experiment_id"] == exp.id
        assert kwargs["researcher_id"] == exp.researcher_id
        assert secagg_arguments == secagg.train_arguments.return_value


class TestGetColumnStats:
    def test_target_dropped(self, module: ModuleType, analytics: MagicMock):
        columns, mean, std, excluded_columns, target_mean = module._get_column_stats(
            analytics, NODE_DATA, "nb:Age"
        )
        assert columns == ["Left-Hippocampus", "Left-Thalamus", "nb:Sex"]
        assert target_mean == 30.0
        assert mean == [4000.0, 7000.0, 0.5]
        assert std == [100.0, 50.0, 0.5]
        assert not excluded_columns
        analytics.fetch_stats.assert_called_once_with(
            ["mean", "std"],
            dataset_schema=["Left-Hippocampus", "Left-Thalamus", "nb:Age", "nb:Sex"],
        )

    def test_excluded_columns(self, module: ModuleType, analytics: MagicMock):
        stats = {
            "mean": GLOBAL_STATS["mean"]
            | {
                "Left-Caudate": 0.0,
                "Right-Putamen": 1.0,
                "lhSurfaceHoles": 20.0,
            },
            "std": GLOBAL_STATS["std"]
            | {
                "Left-Caudate": 0.0,
                "Right-Putamen": np.nan,
                "lhSurfaceHoles": 5.0,
            },
        }
        analytics.fetch_stats.return_value.global_stats.side_effect = stats.get
        node_data = {
            node: {"dtypes": dict.fromkeys(stats["mean"], "Float64")}
            for node in ["NODE_1", "NODE_2"]
        }
        # only in the dataset of one node
        node_data["NODE_1"]["dtypes"]["Left-Amygdala"] = "Float64"
        columns, mean, std, excluded_columns, _ = module._get_column_stats(
            analytics, node_data, "nb:Age"
        )
        assert columns == ["Left-Hippocampus", "Left-Thalamus", "nb:Sex"]
        assert len(mean) == len(std) == 3
        assert (
            "lhSurfaceHoles"
            not in (analytics.fetch_stats.call_args.kwargs["dataset_schema"])
        )
        assert excluded_columns == {
            "not in the datasets of all nodes": ["Left-Amygdala"],
            "not in feature column list": ["lhSurfaceHoles"],
            "zero or infinite standard deviation": [
                "Left-Caudate",
                "Right-Putamen",
            ],
        }

    def test_json_serializable(self, module: ModuleType, analytics: MagicMock):
        json.dumps(module._get_column_stats(analytics, NODE_DATA, "nb:Sex"))

    def test_no_nodes(self, module: ModuleType, analytics: MagicMock):
        with pytest.raises(ValueError, match="No node has a dataset"):
            module._get_column_stats(analytics, {}, "nb:Age")
        analytics.fetch_stats.assert_not_called()

    def test_target_not_in_all_nodes(self, module: ModuleType, analytics: MagicMock):
        node_data = {
            "NODE_1": {"dtypes": {"nb:Age": "Float64", "nb:Sex": "Int64"}},
            "NODE_2": {"dtypes": {"nb:Sex": "Int64"}},
        }
        with pytest.raises(ValueError, match="not in the datasets of all nodes"):
            module._get_column_stats(analytics, node_data, "nb:Age")
        analytics.fetch_stats.assert_not_called()


class TestBuildModelArgs:
    @pytest.mark.parametrize(
        "task,target_column,expected_x",
        [
            ("age", "nb:Age", [[1.0, -1.0, -1.0], [0.0, 0.0, 1.0], [-1.0, 1.0, 1.0]]),
            ("sex", "nb:Sex", [[1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 1.0, 1.0]]),
        ],
    )
    def test_training_data(
        self,
        module: ModuleType,
        analytics: MagicMock,
        fpath_csv: Path,
        task: str,
        target_column: str,
        expected_x: list[list[float]],
    ):
        config = module.TASKS[task]
        columns, mean, std, _, target_mean = module._get_column_stats(
            analytics, NODE_DATA, target_column
        )
        model_args = module._build_model_args(
            task, columns, mean, std, 0.1, target_mean
        )

        plan = config.training_plan_class()
        plan.post_init(model_args, TrainingArgs({}, only_required=False))
        data_manager = plan.training_data()
        data_manager.load(tp_type=plan.type())
        data_manager.load_dataset(root=str(fpath_csv))
        dataset = data_manager._dataset
        np.testing.assert_allclose(
            [dataset[i][0] for i in range(len(dataset))], expected_x
        )

    def test_extra_model_args(self, module: ModuleType):
        model_args = module._build_model_args("sex", ["a"], [0.0], [1.0], 0.1, 0.5)
        assert model_args["n_classes"] == 2
        assert model_args["alpha"] == 0.1
        assert "eta0" not in model_args
        assert "target_mean" not in model_args

    def test_target_mean(self, module: ModuleType):
        model_args = module._build_model_args("age", ["a"], [0.0], [1.0], 0.1, 30.0)
        assert model_args["target_mean"] == 30.0

    @pytest.mark.parametrize(
        "task,expected_metrics",
        [
            ("age", ["MEAN_SQUARE_ERROR", "MEAN_ABSOLUTE_ERROR", "R2"]),
            ("sex", ["ACCURACY", "BALANCED_ACCURACY", "AUROC"]),
        ],
    )
    def test_metrics(self, module: ModuleType, task: str, expected_metrics: list):
        model_args = module._build_model_args(task, ["a"], [0.0], [1.0], 0.1, 30.0)
        assert model_args["metrics"] == expected_metrics


class TestBuildOptimizerArgs:
    @pytest.mark.parametrize(
        "aggregator,expected",
        [
            ("fedavg", {"lr": 0.01, "scaffold": False, "fedprox_mu": None}),
            ("fedprox", {"lr": 0.01, "scaffold": False, "fedprox_mu": 2.0}),
            ("scaffold", {"lr": 0.01, "scaffold": True, "fedprox_mu": None}),
        ],
    )
    def test_build_optimizer_args(
        self, module: ModuleType, aggregator: str, expected: dict
    ):
        assert module._build_optimizer_args(0.01, aggregator, 2.0) == expected


class TestGetAggOptimizer:
    def test_scaffold(self, module: ModuleType):
        optimizer = module._get_agg_optimizer("scaffold")
        assert optimizer.get_state()["config"]["modules"] == [("scaffold-server", {})]

    @pytest.mark.parametrize("aggregator", ["fedavg", "fedprox"])
    def test_other(self, module: ModuleType, aggregator: str):
        assert module._get_agg_optimizer(aggregator) is None


class TestMoveTensorboardLogs:
    def test_move(
        self, module: ModuleType, dpath_fbm_tensorboard: Path, tmp_path: Path
    ):
        exp = MagicMock(tensorboard_results_path=str(dpath_fbm_tensorboard))
        dpath_tensorboard = tmp_path / "age" / "tensorboard"
        module._move_tensorboard_logs(exp, dpath_tensorboard)
        exp.monitor.return_value.close_writer.assert_called_once()
        assert sorted(p.name for p in dpath_tensorboard.iterdir()) == [
            "NODE_1",
            "NODE_2",
        ]
        assert (dpath_tensorboard / "NODE_1" / "events.out.tfevents").exists()
        assert not any(dpath_fbm_tensorboard.iterdir())

    def test_overwrite_node_dir(
        self, module: ModuleType, dpath_fbm_tensorboard: Path, tmp_path: Path
    ):
        exp = MagicMock(tensorboard_results_path=str(dpath_fbm_tensorboard))
        dpath_tensorboard = tmp_path / "tensorboard"
        (dpath_tensorboard / "NODE_1").mkdir(parents=True)
        (dpath_tensorboard / "NODE_1" / "old").touch()
        module._move_tensorboard_logs(exp, dpath_tensorboard)
        assert [p.name for p in (dpath_tensorboard / "NODE_1").iterdir()] == [
            "events.out.tfevents"
        ]


class TestRunTask:
    def test_run_task(
        self, module: ModuleType, mock_experiment: MagicMock, tmp_path: Path
    ):
        fpath_model = tmp_path / "age" / "model.pkl"
        with module.log_to_file(tmp_path / "logs.log"):
            folds = list(
                module._run_task(
                    "age",
                    n_rounds=3,
                    batch_size=8,
                    n_updates=10,
                    learning_rate=0.1,
                    test_ratio=0.2,
                    aggregator="scaffold",
                    fedprox_mu=1.0,
                    alpha=0.1,
                    fpath_model=fpath_model,
                )
            )

        assert mock_experiment.call_args.kwargs["tags"] == ["brainflex", "age"]
        assert mock_experiment.call_args.kwargs["round_limit"] == 3
        assert mock_experiment.call_args.kwargs["tensorboard"] is False
        training_args = TrainingArgs(mock_experiment.call_args.kwargs["training_args"])
        assert training_args["loader_args"] == {"batch_size": 8}
        assert training_args["num_updates"] == 10
        assert training_args["optimizer_args"] == {
            "lr": 0.1,
            "scaffold": True,
            "fedprox_mu": None,
        }
        assert mock_experiment.call_args.kwargs["agg_optimizer"] is not None
        secagg = mock_experiment.call_args.kwargs["secagg"]
        assert secagg.active is True
        assert secagg.clipping_range == 100
        exp = mock_experiment.return_value
        assert exp.run_once.call_count == 3
        assert exp.save_training_replies.call_count == 3
        assert training_args["random_seed"] == module.DEFAULT_RANDOM_SEED
        # different seed at each round
        assert [
            call.args[0]["random_seed"] for call in exp.set_training_args.call_args_list
        ] == [module.DEFAULT_RANDOM_SEED + round_ for round_ in (1, 2, 3)]
        exp.training_plan.return_value.export_model.assert_called_once_with(
            str(fpath_model)
        )
        assert (
            exp.set_training_plan_class.call_args.args[0].__name__
            == "TabularRegressorTrainingPlan"
        )
        log_text = (tmp_path / "logs.log").read_text()
        assert "INFO     - ===== AGE =====" in log_text
        assert "INFO     - Using 3 features" in log_text

        # before training and after each round
        assert len(folds) == 4
        assert {fold_key for fold_key, _ in folds} == {"NODE_1__NODE_2__NODE_3"}
        fold = json.loads(
            json.dumps(folds[-1][1], default=sys.modules["_io_utils"].serialize)
        )
        assert fold["settings"]["excluded_columns"] == {}
        model_args = fold["settings"]["model_args"]
        assert model_args["target_column"] == "nb:Age"
        assert model_args["alpha"] == 0.1
        assert model_args["target_mean"] == 30.0
        assert model_args["feature_columns"] == [
            "Left-Hippocampus",
            "Left-Thalamus",
            "nb:Sex",
        ]
        exp.set_model_args.assert_called_once_with(model_args)
        training_args = fold["settings"]["training_args"]
        # the metrics are computed by the training plan testing_step
        assert "test_metric" not in training_args
        # JSON object keys are strings
        assert fold["results"] == json.loads(
            json.dumps(
                {"node_metrics": NODE_METRICS, "training_replies": TRAINING_REPLIES}
            )
        )

    def test_settings_saved_if_training_fails(
        self, module: ModuleType, mock_experiment: MagicMock, tmp_path: Path
    ):
        mock_experiment.return_value.run_once.side_effect = RuntimeError
        folds = module._run_task(
            "sex",
            n_rounds=3,
            batch_size=8,
            n_updates=10,
            learning_rate=0.1,
            test_ratio=0.2,
            aggregator="fedavg",
            fedprox_mu=1.0,
            alpha=0.1,
            fpath_model=tmp_path / "sex" / "model.pkl",
        )
        # yielded before training so that the settings can be saved
        fold_key, fold = next(folds)
        assert fold_key == "NODE_1__NODE_2__NODE_3"
        assert fold["settings"]["model_args"]["n_classes"] == 2
        assert mock_experiment.call_args.kwargs["agg_optimizer"] is None
        with pytest.raises(RuntimeError):
            next(folds)
        assert "results" not in fold

    def test_tensorboard(
        self, module: ModuleType, mock_experiment: MagicMock, tmp_path: Path
    ):
        dpath_tensorboard = tmp_path / "age" / "tensorboard"
        list(
            module._run_task(
                "age",
                n_rounds=1,
                batch_size=8,
                n_updates=10,
                learning_rate=0.1,
                test_ratio=0.2,
                aggregator="scaffold",
                fedprox_mu=1.0,
                alpha=0.1,
                fpath_model=tmp_path / "age" / "model.pkl",
                dpath_tensorboard=dpath_tensorboard,
            )
        )
        assert mock_experiment.call_args.kwargs["tensorboard"] is True
        assert (dpath_tensorboard / "NODE_2").is_dir()

    def test_no_secagg(
        self,
        module: ModuleType,
        mock_experiment: MagicMock,
        mock_evaluate_held_out: MagicMock,
        mock_setup_held_out_secagg: MagicMock,
        tmp_path: Path,
    ):
        list(
            module._run_task(
                "sex",
                n_rounds=1,
                batch_size=8,
                n_updates=10,
                learning_rate=0.1,
                test_ratio=0.2,
                aggregator="fedavg",
                fedprox_mu=1.0,
                alpha=0.1,
                fpath_model=tmp_path / "model.pkl",
                training_nodes=["NODE_1", "NODE_2"],
                node_data_held_out={"NODE_3": NODE_DATA["NODE_3"]},
                secagg=False,
            )
        )
        secagg = mock_experiment.call_args.kwargs["secagg"]
        assert secagg.active is False
        # default clipping range
        assert secagg.clipping_range is None
        mock_setup_held_out_secagg.assert_not_called()
        mock_evaluate_held_out.assert_called_with(
            mock_experiment.return_value, {"NODE_3": NODE_DATA["NODE_3"]}, None
        )

    @pytest.mark.parametrize(("task", "n_warnings"), [("age", 2), ("sex", 1)])
    def test_leakage_warnings(
        self,
        module: ModuleType,
        mock_experiment: MagicMock,
        mock_evaluate_held_out: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        task: str,
        n_warnings: int,
    ):
        with module.log_to_file(tmp_path / "logs.log"):
            list(
                module._run_task(
                    task,
                    n_rounds=1,
                    batch_size=8,
                    n_updates=10,
                    learning_rate=0.1,
                    test_ratio=0.2,
                    aggregator="fedavg",
                    fedprox_mu=1.0,
                    alpha=0.1,
                    fpath_model=tmp_path / "model.pkl",
                )
            )
        assert [
            record.levelno
            for record in caplog.records
            if "data leakage" in record.getMessage()
        ] == [logging.WARNING] * n_warnings
        assert "data leakage" in (tmp_path / "logs.log").read_text()
        assert mock_experiment.call_args.kwargs["nodes"] is None
        mock_experiment.return_value.set_training_data.assert_not_called()
        training_args = mock_experiment.call_args.kwargs["training_args"]
        assert training_args["shuffle_testing_dataset"] is False
        mock_evaluate_held_out.assert_not_called()

    def test_held_out_node(
        self,
        module: ModuleType,
        mock_experiment: MagicMock,
        mock_evaluate_held_out: MagicMock,
        mock_setup_held_out_secagg: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        node_data_held_out = {"NODE_3": NODE_DATA["NODE_3"]}
        # the held-out node is dropped by the new search
        exp = mock_experiment.return_value
        exp.training_data.return_value.data.return_value = {
            node: NODE_DATA[node] for node in ["NODE_1", "NODE_2"]
        }
        folds = list(
            module._run_task(
                "age",
                n_rounds=3,
                batch_size=8,
                n_updates=10,
                learning_rate=0.1,
                test_ratio=0.2,
                aggregator="scaffold",
                fedprox_mu=1.0,
                alpha=0.1,
                fpath_model=tmp_path / "model.pkl",
                training_nodes=["NODE_1", "NODE_2"],
                node_data_held_out=node_data_held_out,
            )
        )
        assert mock_experiment.call_args.kwargs["nodes"] == ["NODE_1", "NODE_2"]
        mock_experiment.return_value.set_training_data.assert_called_once_with(
            None, from_tags=True
        )
        training_args = mock_experiment.call_args.kwargs["training_args"]
        assert training_args["shuffle_testing_dataset"] is True
        exp = mock_experiment.return_value
        # before each round and after the last one
        assert mock_evaluate_held_out.call_count == 4
        # all the nodes, so that the context has at least 2 nodes
        mock_setup_held_out_secagg.assert_called_once_with(
            exp, ["NODE_1", "NODE_2", "NODE_3"]
        )
        mock_evaluate_held_out.assert_called_with(
            exp, node_data_held_out, mock_setup_held_out_secagg.return_value
        )
        # before training, after each round, and after the final evaluation
        assert len(folds) == 5
        fold_key, fold = folds[-1]
        assert fold_key == "NODE_1__NODE_2"
        assert fold["settings"]["training_nodes"] == ["NODE_1", "NODE_2"]
        assert fold["settings"]["held_out_node"] == "NODE_3"
        assert "node_metrics" in fold["results"]
        assert not any("data leakage" in r.getMessage() for r in caplog.records)
        assert "===== AGE: held_out_node=NODE_3 =====" in caplog.messages


def _read_results(dpath_task: Path) -> tuple[Path, dict]:
    """Read the only results file of a task."""
    (fpath,) = dpath_task.glob("results-*.json")
    return fpath, json.loads(fpath.read_text())


RESULTS_WITH_FOLD = {
    "settings": {"runtime_args": {}},
    "folds": {"NODE_1__NODE_2": {"settings": {}, "results": {}}},
}


def _write_results(
    module: ModuleType,
    dpath: Path,
    task: str,
    data: dict,
    cv: bool = False,
    date_tag: str | None = None,
) -> None:
    """Write a results file for the default CLI arguments."""
    fpath = module.PathHelper(
        output_dir=dpath,
        date_tag=date_tag if date_tag is not None else module.DEFAULT_DATE_TAG,
        n_rounds=module.DEFAULT_N_ROUNDS,
        batch_size=module.DEFAULT_BATCH_SIZE,
        n_updates=module.DEFAULT_N_UPDATES,
        learning_rate=module.DEFAULT_LEARNING_RATE,
        test_ratio=module.DEFAULT_TEST_RATIO,
        aggregator=module.DEFAULT_AGGREGATOR,
        fedprox_mu=module.DEFAULT_FEDPROX_MU,
        alpha=module.DEFAULT_ALPHA,
        secagg=module.DEFAULT_SECAGG,
        cv=cv,
        random_seed=module.DEFAULT_RANDOM_SEED,
    ).get_fpath_out(task, module.PREFIX_RESULTS)
    fpath.parent.mkdir(parents=True)
    fpath.write_text(json.dumps(data))


class TestMain:
    def test_tasks_in_order(
        self, module: ModuleType, mock_run_task: MagicMock, tmp_path: Path
    ):
        module.main(
            [
                str(tmp_path),
                "--task",
                "sex",
                "--task",
                "age",
                "--n-rounds",
                "2",
                "--no-cv",
            ]
        )
        assert [call.kwargs["task"] for call in mock_run_task.call_args_list] == [
            "sex",
            "age",
        ]
        kwargs = mock_run_task.call_args.kwargs
        assert kwargs["n_rounds"] == 2
        dpath_date = tmp_path.resolve() / module.DEFAULT_DATE_TAG
        fpath_results, json_data = _read_results(dpath_date / "age")
        assert fpath_results.name.startswith("results-scaffold-2rounds-")
        assert kwargs["fpath_model"].name.startswith("model-scaffold-2rounds-")
        assert kwargs["fpath_model"].suffix == ".pkl"
        # created by log_to_file even though _run_task is mocked
        fpath_log = dpath_date / "age" / fpath_results.name.replace("results-", "logs-")
        assert fpath_log.with_suffix(".log").exists()
        assert kwargs["dpath_tensorboard"].parent == dpath_date / "age"
        assert kwargs["dpath_tensorboard"].name.startswith(
            "tensorboard-scaffold-2rounds-"
        )
        assert json_data["folds"] == {
            "NODE_1__NODE_2__NODE_3": {
                "settings": {"training_nodes": None},
                "results": {"node_metrics": {}},
            }
        }
        settings = json_data["settings"]["runtime_args"]
        assert settings["output_dir"] == str(tmp_path.resolve())
        assert settings["tasks"] == ["sex", "age"]
        assert settings["date_tag"] == module.DEFAULT_DATE_TAG
        assert settings["n_rounds"] == 2
        assert settings["learning_rate"] == module.DEFAULT_LEARNING_RATE
        assert settings["overwrite"] is False
        assert settings["tensorboard"] is True
        assert settings["secagg"] is True
        assert kwargs["secagg"] is True

    def test_messages_printed_at_end(
        self,
        module: ModuleType,
        mock_run_task: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ):
        def run_task(task: str, **kwargs) -> Generator[tuple[str, dict], None, None]:
            module.logger.info(f"[{task}] done")
            assert capsys.readouterr().out == ""
            if task == "age":
                raise RuntimeError
            yield from _fake_run_task(**kwargs)

        mock_run_task.side_effect = run_task
        with pytest.raises(RuntimeError):
            module.main([str(tmp_path), "--task", "sex", "--task", "age", "--no-cv"])
        lines = capsys.readouterr().out.splitlines()
        assert lines[0] == "INFO     - [sex] done"
        assert lines[1].startswith("INFO     - Saved settings and results to ")
        assert lines[2].startswith("INFO     - Saved log to ")
        # no "Saved" messages for the failed run
        assert lines[3:] == ["INFO     - [age] done"]

    def test_date_tag(
        self, module: ModuleType, mock_run_task: MagicMock, tmp_path: Path
    ):
        module.main([str(tmp_path), "--task", "age", "--date-tag", "mytag", "--no-cv"])
        _, json_data = _read_results(tmp_path.resolve() / "mytag" / "age")
        assert json_data["settings"]["runtime_args"]["date_tag"] == "mytag"

    def test_random_seed(
        self, module: ModuleType, mock_run_task: MagicMock, tmp_path: Path
    ):
        module.main([str(tmp_path), "--task", "age", "--random-seed", "7", "--no-cv"])
        kwargs = mock_run_task.call_args.kwargs
        assert kwargs["random_seed"] == 7
        assert "-7seed-" in kwargs["fpath_model"].name
        fpath_results, json_data = _read_results(
            tmp_path.resolve() / module.DEFAULT_DATE_TAG / "age"
        )
        assert "-7seed-" in fpath_results.name
        assert json_data["settings"]["runtime_args"]["random_seed"] == 7

    def test_default_random_seed(
        self, module: ModuleType, mock_run_task: MagicMock, tmp_path: Path
    ):
        module.main([str(tmp_path), "--task", "age", "--no-cv"])
        assert mock_run_task.call_args.kwargs["random_seed"] == 3791

    def test_no_secagg(
        self, module: ModuleType, mock_run_task: MagicMock, tmp_path: Path
    ):
        module.main([str(tmp_path), "--task", "age", "--no-secagg", "--no-cv"])
        assert mock_run_task.call_args.kwargs["secagg"] is False
        _, json_data = _read_results(
            tmp_path.resolve() / module.DEFAULT_DATE_TAG / "age"
        )
        assert json_data["settings"]["runtime_args"]["secagg"] is False

    def test_aggregator_options(
        self, module: ModuleType, mock_run_task: MagicMock, tmp_path: Path
    ):
        module.main(
            [
                str(tmp_path),
                "--task",
                "age",
                "--aggregator",
                "fedprox",
                "--fedprox-mu",
                "5",
                "--alpha",
                "0.5",
                "--no-cv",
            ]
        )
        kwargs = mock_run_task.call_args.kwargs
        assert kwargs["aggregator"] == "fedprox"
        assert kwargs["fedprox_mu"] == 5.0
        assert kwargs["alpha"] == 0.5
        fpath_results, _ = _read_results(
            tmp_path.resolve() / module.DEFAULT_DATE_TAG / "age"
        )
        assert fpath_results.name.startswith("results-fedprox_5.0mu-")
        assert "-0.5alpha-" in fpath_results.stem

    def test_no_tensorboard(
        self, module: ModuleType, mock_run_task: MagicMock, tmp_path: Path
    ):
        module.main([str(tmp_path), "--task", "age", "--no-tensorboard", "--no-cv"])
        assert mock_run_task.call_args.kwargs["dpath_tensorboard"] is None
        _, json_data = _read_results(
            tmp_path.resolve() / module.DEFAULT_DATE_TAG / "age"
        )
        assert json_data["settings"]["runtime_args"]["tensorboard"] is False

    def test_existing_results(
        self, module: ModuleType, mock_run_task: MagicMock, tmp_path: Path
    ):
        _write_results(module, tmp_path.resolve(), "age", RESULTS_WITH_FOLD)
        with pytest.raises(SystemExit, match="--overwrite"):
            module.main([str(tmp_path), "--task", "sex", "--task", "age", "--no-cv"])
        mock_run_task.assert_not_called()

    def test_existing_settings_only(
        self, module: ModuleType, mock_run_task: MagicMock, tmp_path: Path
    ):
        # a run that failed before any results were saved can be rerun
        _write_results(
            module,
            tmp_path.resolve(),
            "age",
            {"settings": {}, "folds": {"NODE_1__NODE_2": {"settings": {}}}},
        )
        module.main([str(tmp_path), "--task", "age", "--no-cv"])
        mock_run_task.assert_called_once()

    def test_existing_results_other_date_tag(
        self, module: ModuleType, mock_run_task: MagicMock, tmp_path: Path
    ):
        _write_results(
            module, tmp_path.resolve(), "age", RESULTS_WITH_FOLD, date_tag="old"
        )
        module.main([str(tmp_path), "--task", "age", "--no-cv"])
        mock_run_task.assert_called_once()

    def test_overwrite(
        self, module: ModuleType, mock_run_task: MagicMock, tmp_path: Path
    ):
        _write_results(module, tmp_path.resolve(), "age", RESULTS_WITH_FOLD)
        module.main([str(tmp_path), "--task", "age", "--overwrite", "--no-cv"])
        mock_run_task.assert_called_once()

    def test_cv(
        self,
        module: ModuleType,
        mock_run_task: MagicMock,
        mock_get_node_data: MagicMock,
        tmp_path: Path,
    ):
        # CV is on by default
        module.main([str(tmp_path), "--task", "sex", "--task", "age"])
        calls = [call.kwargs for call in mock_run_task.call_args_list]
        assert [
            (kwargs["task"], *kwargs["node_data_held_out"]) for kwargs in calls
        ] == [(task, node) for task in ["sex", "age"] for node in NODE_DATA]
        kwargs = calls[0]
        assert kwargs["training_nodes"] == ["NODE_2", "NODE_3"]
        assert kwargs["node_data_held_out"] == {"NODE_1": NODE_DATA["NODE_1"]}
        assert kwargs["fpath_model"].stem.endswith("-heldout_NODE_1")
        assert kwargs["dpath_tensorboard"].name.endswith("-heldout_NODE_1")
        # one results file and one log file per task, with all the folds
        dpath_date = tmp_path.resolve() / module.DEFAULT_DATE_TAG
        for task in ["sex", "age"]:
            fpath_results, json_data = _read_results(dpath_date / task)
            assert fpath_results.name.endswith("-secagg-cv.json")
            assert json_data["settings"]["runtime_args"]["cv"] is True
            assert list(json_data["folds"]) == [
                "NODE_2__NODE_3",
                "NODE_1__NODE_3",
                "NODE_1__NODE_2",
            ]
            assert all("results" in fold for fold in json_data["folds"].values())
            assert [p.name for p in (dpath_date / task).glob("logs-*")] == [
                fpath_results.name.replace("results-", "logs-").replace(".json", ".log")
            ]

    def test_no_cv(
        self,
        module: ModuleType,
        mock_run_task: MagicMock,
        mock_get_node_data: MagicMock,
        tmp_path: Path,
    ):
        module.main([str(tmp_path), "--task", "age", "--no-cv"])
        mock_get_node_data.assert_not_called()
        kwargs = mock_run_task.call_args.kwargs
        assert kwargs["training_nodes"] is None
        assert kwargs["node_data_held_out"] is None

    @pytest.mark.parametrize(
        ("secagg", "n_nodes", "match"),
        [
            (False, 1, "at least 2 nodes, "),
            (True, 2, "at least 3 nodes with secure aggregation"),
        ],
    )
    def test_cv_not_enough_nodes(
        self,
        module: ModuleType,
        mock_run_task: MagicMock,
        mock_get_node_data: MagicMock,
        tmp_path: Path,
        secagg: bool,
        n_nodes: int,
        match: str,
    ):
        mock_get_node_data.return_value = dict(list(NODE_DATA.items())[:n_nodes])
        with pytest.raises(ValueError, match=match):
            module.run_experiments(tmp_path, ["age"], secagg=secagg)
        mock_run_task.assert_not_called()

    def test_cv_two_nodes_no_secagg(
        self,
        module: ModuleType,
        mock_run_task: MagicMock,
        mock_get_node_data: MagicMock,
        tmp_path: Path,
    ):
        mock_get_node_data.return_value = dict(list(NODE_DATA.items())[:2])
        module.run_experiments(tmp_path, ["age"], secagg=False)
        assert mock_run_task.call_count == 2

    def test_cv_existing_results(
        self,
        module: ModuleType,
        mock_run_task: MagicMock,
        mock_get_node_data: MagicMock,
        tmp_path: Path,
    ):
        _write_results(module, tmp_path.resolve(), "age", RESULTS_WITH_FOLD, cv=True)
        with pytest.raises(SystemExit, match="-cv.json"):
            module.main([str(tmp_path), "--task", "age"])
        mock_run_task.assert_not_called()

    def test_default_tasks(
        self, module: ModuleType, mock_run_task: MagicMock, tmp_path: Path
    ):
        module.main([str(tmp_path), "--no-cv"])
        assert [
            call.kwargs["task"] for call in mock_run_task.call_args_list
        ] == module.DEFAULT_TASKS

    def test_invalid_task(
        self, module: ModuleType, mock_run_task: MagicMock, tmp_path: Path
    ):
        with pytest.raises(SystemExit):
            module.main([str(tmp_path), "--task", "height"])
        mock_run_task.assert_not_called()
