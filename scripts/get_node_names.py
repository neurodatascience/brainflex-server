#!/usr/bin/env python3

"""Print the node name and other identifying fields of every connected node."""

from fedbiomed.common.message import ListRequest
from fedbiomed.researcher.federated_workflows import Experiment


def main() -> None:
    exp = Experiment()  # no tags
    message = ListRequest(researcher_id=exp.researcher_id)

    with exp.requests.send(message) as federated_req:
        replies = federated_req.replies()

    if not replies:
        print("No node replied")
        return

    for node_id, reply in replies.items():
        fields = reply.get_dict() if hasattr(reply, "get_dict") else vars(reply)
        print(f"\n=== {node_id} ===")
        for key, value in fields.items():
            if key == "databases":
                print(f"  {key}: {len(value)} dataset(s)")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
