#!/usr/bin/env python3

"""Request the approval of the training plans by the nodes."""

import argparse
import sys

from fedbiomed.common.constants import TrainingPlanApprovalStatus
from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.federated_workflows.jobs import (
    TrainingPlanApproveJob,
    TrainingPlanCheckJob,
)

# importable because the script's directory is the first entry of sys.path
from run_experiments import COMMON_TAG, TASKS, _load_training_plan_class

# Fed-BioMed-defined status of a training plan that a node doesn't know
TRAINING_PLAN_NOT_REGISTERED = "Not Registered"
STATUS_SENT = "sent for approval"


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--task",
        dest="tasks",
        action="append",
        choices=list(TASKS),
        help="Task whose training plan to send. Can be repeated. Default: all tasks.",
    )
    return parser


def _request_approval(task: str) -> dict[str, str]:
    """Check the approval of a task's training plan by the nodes with data for it.

    Returns the status of each node that has not approved the training plan.
    """
    config = TASKS[task]
    exp = Experiment(tags=[COMMON_TAG, task])
    fds = exp.training_data()
    nodes = fds.node_ids() if fds is not None else []
    if not nodes:
        raise ValueError(f"No node has a dataset for {task}")
    training_plan = _load_training_plan_class(
        config.training_plan_module, config.training_plan_class
    )()

    replies = TrainingPlanCheckJob(
        researcher_id=exp.researcher_id,
        requests=exp.requests,
        nodes=nodes,
        experiment_id=exp.id,
        training_plan=training_plan,
    ).execute()
    not_approved: dict[str, str] = {}
    for node in nodes:
        reply = replies.get(node)
        if reply is None:
            not_approved[node] = "no reply"
        elif not reply.success:
            not_approved[node] = reply.msg
        elif (
            reply.approval_obligation
            and reply.status != TrainingPlanApprovalStatus.APPROVED.value
        ):
            not_approved[node] = reply.status

    if nodes_not_registered := [
        node
        for node, status in not_approved.items()
        if status == TRAINING_PLAN_NOT_REGISTERED
    ]:
        TrainingPlanApproveJob(
            researcher_id=exp.researcher_id,
            requests=exp.requests,
            nodes=nodes_not_registered,
            training_plan=training_plan,
            description=f"{COMMON_TAG} {task}: {config.training_plan_class}",
        ).execute()
        for node in nodes_not_registered:
            not_approved[node] = STATUS_SENT
    return not_approved


def main(argv: list[str] | None = None) -> None:
    args = _get_parser().parse_args(argv)
    all_approved = True
    for task in args.tasks or list(TASKS):
        try:
            not_approved = _request_approval(task)
        except ValueError as e:
            print(f"[{task}] Error: {e}")
            all_approved = False
            continue
        if not_approved:
            print(f"[{task}] Training plan not approved by all nodes: {not_approved}")
            all_approved = False
        else:
            print(f"[{task}] Training plan approved by all nodes")
    if not all_approved:
        print()
        sys.exit(
            "Error: the training plans are not approved by all nodes. Ask nodes to "
            "approve them, then run this script again to check"
        )


if __name__ == "__main__":
    main()
