import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from fedbiomed.common.constants import TrainingPlanApprovalStatus
from fedbiomed.common.training_plans import BaseTrainingPlan
from pytest_mock import MockerFixture

FPATH_SCRIPT = (
    Path(__file__).parents[2] / "scripts" / "request_training_plan_approval.py"
)
NODES = ["NODE_1", "NODE_2", "NODE_3", "NODE_4"]


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
    # as when the script is run
    monkeypatch.syspath_prepend(str(FPATH_SCRIPT.parent))
    spec = importlib.util.spec_from_file_location(
        "request_training_plan_approval", FPATH_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mock_experiment(module: ModuleType, mocker: MockerFixture) -> MagicMock:
    mock_experiment = mocker.patch.object(module, "Experiment")
    mock_experiment.return_value.training_data.return_value.node_ids.return_value = (
        NODES
    )
    return mock_experiment


@pytest.fixture
def mock_check_job(module: ModuleType, mocker: MockerFixture) -> MagicMock:
    return mocker.patch.object(module, "TrainingPlanCheckJob")


@pytest.fixture
def mock_approve_job(module: ModuleType, mocker: MockerFixture) -> MagicMock:
    return mocker.patch.object(module, "TrainingPlanApproveJob")


def _status_reply(
    status: str = TrainingPlanApprovalStatus.APPROVED.value,
    success: bool = True,
    approval_obligation: bool = True,
) -> MagicMock:
    return MagicMock(
        status=status,
        success=success,
        approval_obligation=approval_obligation,
        msg="message",
    )


@pytest.mark.usefixtures("mock_experiment")
class TestRequestApproval:
    @pytest.mark.parametrize("task", ["age", "sex"])
    def test_approved(
        self,
        module: ModuleType,
        mock_experiment: MagicMock,
        mock_check_job: MagicMock,
        mock_approve_job: MagicMock,
        task: str,
    ):
        mock_check_job.return_value.execute.return_value = {
            "NODE_1": _status_reply(),
            "NODE_2": _status_reply(),
            "NODE_3": _status_reply(),
            # approval not required
            "NODE_4": _status_reply(status="Pending", approval_obligation=False),
        }
        assert module._request_approval(task) == {}
        assert mock_experiment.call_args.kwargs["tags"] == [module.COMMON_TAG, task]
        kwargs = mock_check_job.call_args.kwargs
        assert kwargs["nodes"] == NODES
        training_plan = kwargs["training_plan"]
        assert isinstance(training_plan, BaseTrainingPlan)
        assert type(training_plan) is module.TASKS[task].training_plan_class
        mock_approve_job.assert_not_called()

    def test_not_approved(
        self,
        module: ModuleType,
        mock_check_job: MagicMock,
        mock_approve_job: MagicMock,
    ):
        mock_check_job.return_value.execute.return_value = {
            "NODE_1": _status_reply(status=module.TRAINING_PLAN_NOT_REGISTERED),
            "NODE_2": _status_reply(status="Pending"),
            "NODE_3": _status_reply(success=False),
        }
        assert module._request_approval("age") == {
            "NODE_1": module.STATUS_SENT,
            "NODE_2": "Pending",
            "NODE_3": "message",
            "NODE_4": "no reply",
        }
        # only sent to the nodes that don't have the training plan
        kwargs = mock_approve_job.call_args.kwargs
        assert kwargs["nodes"] == ["NODE_1"]
        assert (
            kwargs["training_plan"] == mock_check_job.call_args.kwargs["training_plan"]
        )
        assert kwargs["description"] == "brainflex age: TabularRegressorTrainingPlan"
        mock_approve_job.return_value.execute.assert_called_once_with()

    def test_no_nodes(
        self,
        module: ModuleType,
        mock_experiment: MagicMock,
        mock_check_job: MagicMock,
    ):
        mock_experiment.return_value.training_data.return_value = None
        with pytest.raises(ValueError, match="No node has a dataset for age"):
            module._request_approval("age")
        mock_check_job.assert_not_called()


class TestMain:
    @pytest.fixture
    def mock_request_approval(
        self, module: ModuleType, mocker: MockerFixture
    ) -> MagicMock:
        return mocker.patch.object(module, "_request_approval", return_value={})

    def test_all_tasks(
        self,
        module: ModuleType,
        mock_request_approval: MagicMock,
        capsys: pytest.CaptureFixture,
    ):
        module.main([])
        assert [call.args[0] for call in mock_request_approval.call_args_list] == list(
            module.TASKS
        )
        assert "[age] Training plan approved by all nodes" in capsys.readouterr().out

    def test_task(self, module: ModuleType, mock_request_approval: MagicMock):
        module.main(["--task", "sex"])
        mock_request_approval.assert_called_once_with("sex")

    def test_not_approved(
        self,
        module: ModuleType,
        mock_request_approval: MagicMock,
        capsys: pytest.CaptureFixture,
    ):
        mock_request_approval.side_effect = [
            {"NODE_1": "Pending"},
            ValueError("No node has a dataset for sex"),
        ]
        # all tasks are processed so that the nodes can approve them at once
        with pytest.raises(SystemExit, match="not approved by all nodes"):
            module.main(["--task", "age", "--task", "sex"])
        out = capsys.readouterr().out
        assert (
            "[age] Training plan not approved by all nodes: {'NODE_1': 'Pending'}"
            in out
        )
        assert "[sex] Error: No node has a dataset for sex" in out

    def test_invalid_task(self, module: ModuleType, mock_request_approval: MagicMock):
        with pytest.raises(SystemExit):
            module.main(["--task", "height"])
        mock_request_approval.assert_not_called()
