"""Output paths and JSON helpers for the experiment scripts."""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

PREFIX_MODEL = "model"
PREFIX_RESULTS = "results"
PREFIX_TENSORBOARD = "tensorboard"
PREFIX_LOGS = "logs"


@dataclass(frozen=True)
class PathHelper:
    output_dir: Path
    date_tag: str
    n_rounds: int
    batch_size: int
    n_updates: int
    learning_rate: float
    test_ratio: float
    aggregator: str
    fedprox_mu: float
    alpha: float
    secagg: bool
    cv: bool

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
                (
                    self.aggregator
                    if self.aggregator != "fedprox"
                    else f"{self.aggregator}_{self.fedprox_mu}mu"
                ),
                f"{self.n_rounds}rounds",
                f"{self.n_updates}updates",
                f"{self.batch_size}",
                f"{self.learning_rate}lr",
                f"{self.test_ratio}test",
                f"{self.alpha}alpha",
                "secagg" if self.secagg else "no_secagg",
                "cv" if self.cv else "no_cv",
            ]
        )
        if held_out_node is not None:
            stem = f"{stem}-heldout_{held_out_node}"
        return self.output_dir / self.date_tag / task / f"{stem}{extension}"


def serialize(obj: object) -> object:
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def save_json(fpath: Path, data: dict) -> None:
    fpath.write_text(
        json.dumps(data, indent=4, allow_nan=False, default=serialize) + "\n"
    )
