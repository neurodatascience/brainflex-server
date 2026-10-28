import importlib.util
import json
import sys
import warnings
from enum import Enum
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import BaseTrainingPlan
from pytest_mock import MockerFixture

FPATH_SCRIPT = Path(__file__).parents[2] / "scripts" / "run_experiments.py"
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
    node: {"dataset_id": f"dataset_{node}", "name": f"name_{node}"}
    for node in ["NODE_1", "NODE_2", "NODE_3"]
}


@pytest.fixture
def module(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> ModuleType:
    # importing fedbiomed.researcher creates a component directory
    monkeypatch.setenv(
        "FBM_RESEARCHER_COMPONENT_ROOT",
        str(tmp_path_factory.getbasetemp() / "fbm-researcher"),
    )
    # copy so that modules imported by the script are not kept
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    spec = importlib.util.spec_from_file_location("run_experiments", FPATH_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    # mimic the getters of a real Experiment
    exp.model_args.side_effect = lambda: exp.set_model_args.call_args.args[0]
    exp.training_args.side_effect = lambda: TrainingArgs(
        mock_experiment.call_args.kwargs["training_args"]
    ).dict()
    exp.monitor.return_value._metric_store = NODE_METRICS
    exp.save_training_replies.return_value = TRAINING_REPLIES
    exp.tensorboard_results_path = str(dpath_fbm_tensorboard)
    return mock_experiment


@pytest.fixture
def mock_run_task(module: ModuleType, mocker: MockerFixture) -> MagicMock:
    return mocker.patch.object(module, "_run_task")


@pytest.fixture
def mock_evaluate_held_out(module: ModuleType, mocker: MockerFixture) -> MagicMock:
    return mocker.patch.object(module, "_evaluate_held_out")


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


class TestLoadTrainingPlanClass:
    @pytest.mark.parametrize("task", ["age", "sex"])
    def test_source(self, module: ModuleType, task: str):
        config = module.TASKS[task]
        plan_cls = module._load_training_plan_class(
            config.training_plan_module, config.training_plan_class
        )
        assert issubclass(plan_cls, BaseTrainingPlan)
        assert f"class {config.training_plan_class}" in plan_cls().source()


@pytest.fixture
def path_helper(module: ModuleType, tmp_path: Path) -> object:
    return module.PathHelper(
        output_dir=tmp_path,
        n_rounds=10,
        batch_size=32,
        n_updates=50,
        learning_rate=0.001,
        test_ratio=0.2,
        aggregator="scaffold",
        alpha=0.1,
    )


class TestPathHelper:
    def test_get_fpath_out(self, path_helper: object, tmp_path: Path):
        assert path_helper.get_fpath_out("age", "results") == (
            tmp_path
            / "age"
            / "results-scaffold-10rounds-50updates-32-0.001lr-0.2test-0.1alpha.json"
        )

    def test_extension(self, path_helper: object, tmp_path: Path):
        fpath = path_helper.get_fpath_out("sex", "model", ".pkl")
        assert fpath.parent == tmp_path / "sex"
        assert fpath.suffix == ".pkl"

    def test_held_out_node(self, path_helper: object):
        fpath = path_helper.get_fpath_out(
            "age", "model", ".pkl", held_out_node="NODE_1"
        )
        assert fpath.name.endswith("-0.1alpha-heldout_NODE_1.pkl")


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
        mock_training_job.return_value.execute.assert_called_once_with()


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
        assert module._filter_feature_col(col)

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
        assert not module._filter_feature_col(col)


class TestGetFeatureStats:
    def test_target_dropped(self, module: ModuleType, analytics: MagicMock):
        columns, mean, std, excluded_columns, target_mean = module._get_feature_stats(
            analytics, "nb:Age"
        )
        assert columns == ["Left-Hippocampus", "Left-Thalamus", "nb:Sex"]
        assert target_mean == 30.0
        assert mean == [4000.0, 7000.0, 0.5]
        assert std == [100.0, 50.0, 0.5]
        assert not excluded_columns
        analytics.fetch_stats.assert_called_once_with(["mean", "std"])

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
        columns, mean, std, excluded_columns, _ = module._get_feature_stats(
            analytics, "nb:Age"
        )
        assert columns == ["Left-Hippocampus", "Left-Thalamus", "nb:Sex"]
        assert len(mean) == len(std) == 3
        assert excluded_columns == {
            "not in feature column list": ["lhSurfaceHoles"],
            "zero or infinite standard deviation": [
                "Left-Caudate",
                "Right-Putamen",
            ],
        }

    def test_json_serializable(self, module: ModuleType, analytics: MagicMock):
        json.dumps(module._get_feature_stats(analytics, "nb:Sex"))


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
        columns, mean, std, _, target_mean = module._get_feature_stats(
            analytics, target_column
        )
        model_args = module._build_model_args(
            task, columns, mean, std, 0.1, target_mean
        )

        plan = module._load_training_plan_class(
            config.training_plan_module, config.training_plan_class
        )()
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
        assert module._serialize(obj) == expected

    def test_not_serializable(self, module: ModuleType):
        with pytest.raises(TypeError, match="not serializable"):
            module._serialize(object())


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
    @pytest.mark.filterwarnings("ignore:.*data leakage")
    def test_run_task(
        self, module: ModuleType, mock_experiment: MagicMock, tmp_path: Path
    ):
        settings = {"n_rounds": 3}
        fpath_results = tmp_path / "age" / "results.json"
        fpath_model = tmp_path / "age" / "model.pkl"
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
            settings=settings,
            fpath_results=fpath_results,
            fpath_model=fpath_model,
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
        exp = mock_experiment.return_value
        assert exp.run_once.call_count == 3
        assert exp.save_training_replies.call_count == 3
        exp.training_plan.return_value.export_model.assert_called_once_with(
            str(fpath_model)
        )
        assert (
            exp.set_training_plan_class.call_args.args[0].__name__
            == "TabularRegressorTrainingPlan"
        )

        json_data = json.loads(fpath_results.read_text())
        assert json_data["settings"]["runtime_args"] == settings
        assert json_data["settings"]["excluded_columns"] == {}
        model_args = json_data["settings"]["model_args"]
        assert model_args["target_column"] == "nb:Age"
        assert model_args["alpha"] == 0.1
        assert model_args["target_mean"] == 30.0
        assert model_args["feature_columns"] == [
            "Left-Hippocampus",
            "Left-Thalamus",
            "nb:Sex",
        ]
        exp.set_model_args.assert_called_once_with(model_args)
        training_args = json_data["settings"]["training_args"]
        assert training_args["test_metric"] == "MEAN_SQUARE_ERROR"
        # JSON object keys are strings
        assert json_data["results"] == json.loads(
            json.dumps(
                {"node_metrics": NODE_METRICS, "training_replies": TRAINING_REPLIES}
            )
        )

    @pytest.mark.filterwarnings("ignore:.*data leakage")
    def test_settings_saved_if_training_fails(
        self, module: ModuleType, mock_experiment: MagicMock, tmp_path: Path
    ):
        mock_experiment.return_value.run_once.side_effect = RuntimeError
        fpath_results = tmp_path / "sex" / "results.json"
        with pytest.raises(RuntimeError):
            module._run_task(
                "sex",
                n_rounds=3,
                batch_size=8,
                n_updates=10,
                learning_rate=0.1,
                test_ratio=0.2,
                aggregator="fedavg",
                fedprox_mu=1.0,
                alpha=0.1,
                settings={},
                fpath_results=fpath_results,
                fpath_model=tmp_path / "sex" / "model.pkl",
            )
        json_data = json.loads(fpath_results.read_text())
        assert json_data["settings"]["model_args"]["n_classes"] == 2
        assert mock_experiment.call_args.kwargs["agg_optimizer"] is None
        assert "results" not in json_data

    @pytest.mark.filterwarnings("ignore:.*data leakage")
    def test_tensorboard(
        self, module: ModuleType, mock_experiment: MagicMock, tmp_path: Path
    ):
        dpath_tensorboard = tmp_path / "age" / "tensorboard"
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
            settings={},
            fpath_results=tmp_path / "age" / "results.json",
            fpath_model=tmp_path / "age" / "model.pkl",
            dpath_tensorboard=dpath_tensorboard,
        )
        assert mock_experiment.call_args.kwargs["tensorboard"] is True
        assert (dpath_tensorboard / "NODE_2").is_dir()

    @pytest.mark.parametrize(("task", "n_warnings"), [("age", 2), ("sex", 1)])
    def test_leakage_warnings(
        self,
        module: ModuleType,
        mock_experiment: MagicMock,
        mock_evaluate_held_out: MagicMock,
        tmp_path: Path,
        task: str,
        n_warnings: int,
    ):
        with pytest.warns(UserWarning, match="data leakage") as record:
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
                settings={},
                fpath_results=tmp_path / "results.json",
                fpath_model=tmp_path / "model.pkl",
            )
        assert len(record) == n_warnings
        assert mock_experiment.call_args.kwargs["nodes"] is None
        training_args = mock_experiment.call_args.kwargs["training_args"]
        assert training_args["shuffle_testing_dataset"] is False
        mock_evaluate_held_out.assert_not_called()

    def test_held_out_node(
        self,
        module: ModuleType,
        mock_experiment: MagicMock,
        mock_evaluate_held_out: MagicMock,
        tmp_path: Path,
    ):
        fpath_results = tmp_path / "results.json"
        node_data_held_out = {"NODE_3": NODE_DATA["NODE_3"]}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
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
                settings={},
                fpath_results=fpath_results,
                fpath_model=tmp_path / "model.pkl",
                training_nodes=["NODE_1", "NODE_2"],
                node_data_held_out=node_data_held_out,
            )
        assert mock_experiment.call_args.kwargs["nodes"] == ["NODE_1", "NODE_2"]
        training_args = mock_experiment.call_args.kwargs["training_args"]
        assert training_args["shuffle_testing_dataset"] is True
        exp = mock_experiment.return_value
        # before each round and after the last one
        assert mock_evaluate_held_out.call_count == 4
        mock_evaluate_held_out.assert_called_with(exp, node_data_held_out)
        json_data = json.loads(fpath_results.read_text())
        assert json_data["settings"]["training_nodes"] == ["NODE_1", "NODE_2"]
        assert json_data["settings"]["held_out_node"] == node_data_held_out
        assert "node_metrics" in json_data["results"]


def _write_results(
    module: ModuleType,
    dpath: Path,
    task: str,
    data: dict,
    held_out_node: str | None = None,
) -> None:
    """Write a results file for the default CLI arguments."""
    fpath = module.PathHelper(
        output_dir=dpath,
        n_rounds=module.DEFAULT_N_ROUNDS,
        batch_size=module.DEFAULT_BATCH_SIZE,
        n_updates=module.DEFAULT_N_UPDATES,
        learning_rate=module.DEFAULT_LEARNING_RATE,
        test_ratio=module.DEFAULT_TEST_RATIO,
        aggregator=module.DEFAULT_AGGREGATOR,
        alpha=module.DEFAULT_ALPHA,
    ).get_fpath_out(task, module.PREFIX_RESULTS, held_out_node=held_out_node)
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
        assert kwargs["fpath_results"].parent == tmp_path.resolve() / "age"
        assert kwargs["fpath_results"].name.startswith("results-scaffold-2rounds-")
        assert kwargs["fpath_model"].name.startswith("model-scaffold-2rounds-")
        assert kwargs["fpath_model"].suffix == ".pkl"
        assert kwargs["dpath_tensorboard"].parent == tmp_path.resolve() / "age"
        assert kwargs["dpath_tensorboard"].name.startswith(
            "tensorboard-scaffold-2rounds-"
        )
        settings = kwargs["settings"]
        assert settings["output_dir"] == tmp_path.resolve()
        assert settings["tasks"] == ["sex", "age"]
        assert settings["n_rounds"] == 2
        assert settings["learning_rate"] == module.DEFAULT_LEARNING_RATE
        assert settings["overwrite"] is False
        assert settings["tensorboard"] is True

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
        assert kwargs["fpath_results"].name.startswith("results-fedprox-")
        assert kwargs["fpath_results"].stem.endswith("-0.5alpha")

    def test_no_tensorboard(
        self, module: ModuleType, mock_run_task: MagicMock, tmp_path: Path
    ):
        module.main([str(tmp_path), "--task", "age", "--no-tensorboard", "--no-cv"])
        assert mock_run_task.call_args.kwargs["dpath_tensorboard"] is None
        assert mock_run_task.call_args.kwargs["settings"]["tensorboard"] is False

    def test_existing_results(
        self, module: ModuleType, mock_run_task: MagicMock, tmp_path: Path
    ):
        _write_results(module, tmp_path.resolve(), "age", {"results": {}})
        with pytest.raises(SystemExit, match="--overwrite"):
            module.main([str(tmp_path), "--task", "sex", "--task", "age", "--no-cv"])
        mock_run_task.assert_not_called()

    def test_existing_settings_only(
        self, module: ModuleType, mock_run_task: MagicMock, tmp_path: Path
    ):
        # a run that failed before any results were saved can be rerun
        _write_results(module, tmp_path.resolve(), "age", {"settings": {}})
        module.main([str(tmp_path), "--task", "age", "--no-cv"])
        mock_run_task.assert_called_once()

    def test_overwrite(
        self, module: ModuleType, mock_run_task: MagicMock, tmp_path: Path
    ):
        _write_results(module, tmp_path.resolve(), "age", {"results": {}})
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
        assert kwargs["fpath_results"].stem.endswith("-heldout_NODE_1")
        assert kwargs["fpath_model"].stem.endswith("-heldout_NODE_1")
        assert kwargs["dpath_tensorboard"].name.endswith("-heldout_NODE_1")
        assert kwargs["settings"]["cv"] is True

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

    def test_cv_one_node(
        self,
        module: ModuleType,
        mock_run_task: MagicMock,
        mock_get_node_data: MagicMock,
        tmp_path: Path,
    ):
        mock_get_node_data.return_value = {"NODE_1": NODE_DATA["NODE_1"]}
        with pytest.raises(ValueError, match="at least 2 nodes"):
            module.run_experiments(tmp_path, ["age"])
        mock_run_task.assert_not_called()

    def test_cv_existing_results(
        self,
        module: ModuleType,
        mock_run_task: MagicMock,
        mock_get_node_data: MagicMock,
        tmp_path: Path,
    ):
        _write_results(
            module, tmp_path.resolve(), "age", {"results": {}}, held_out_node="NODE_2"
        )
        with pytest.raises(SystemExit, match="heldout_NODE_2"):
            module.main([str(tmp_path), "--task", "age"])
        mock_run_task.assert_not_called()

    @pytest.mark.parametrize("args", [[], ["--task", "height"]])
    def test_invalid_task(
        self,
        module: ModuleType,
        mock_run_task: MagicMock,
        tmp_path: Path,
        args: list[str],
    ):
        with pytest.raises(SystemExit):
            module.main([str(tmp_path), *args])
        mock_run_task.assert_not_called()
