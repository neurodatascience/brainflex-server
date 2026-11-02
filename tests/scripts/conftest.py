import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

DPATH_SCRIPTS = Path(__file__).parents[2] / "scripts"


@pytest.fixture
def load_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Callable[[str], ModuleType]:
    """Load a module from the scripts directory."""
    # importing fedbiomed.researcher creates a component directory
    monkeypatch.setenv(
        "FBM_RESEARCHER_COMPONENT_ROOT",
        str(tmp_path_factory.getbasetemp() / "fbm-researcher"),
    )
    # copy so that modules imported by the script are not kept
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    # have scripts directory be first in syspath
    monkeypatch.syspath_prepend(str(DPATH_SCRIPTS))

    def _load(name: str) -> ModuleType:
        spec = importlib.util.spec_from_file_location(
            name, DPATH_SCRIPTS / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    return _load
