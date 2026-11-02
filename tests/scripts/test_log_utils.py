import logging
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def module(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("_log_utils")


@pytest.fixture(autouse=True)
def fbm_logger_level(caplog: pytest.LogCaptureFixture) -> None:
    # set by Fed-BioMed's logger, which is not imported here
    caplog.set_level(logging.INFO, logger="fedbiomed")


class TestLogToFile:
    def test_log_to_file(self, module: ModuleType, tmp_path: Path):
        fpath_log = tmp_path / "logs" / "logs.log"
        fbm_logger = logging.getLogger("fedbiomed")
        with module.log_to_file(fpath_log):
            module.logger.info("script info")
            module.logger.warning("script warning")
            fbm_logger.info("fbm info")
            fbm_logger.warning("fbm warning")
            fbm_logger.error("fbm security", extra={"is_security": True})
        lines = fpath_log.read_text().splitlines()
        assert len(lines) == 3
        assert lines[0].endswith("brainflex INFO     - script info")
        assert lines[1].endswith("brainflex WARNING  - script warning")
        assert lines[2].endswith("fedbiomed WARNING  - fbm warning")

    @pytest.mark.parametrize(
        ("node_level", "kept"), [("WARNING", True), ("ERROR", True), ("INFO", False)]
    )
    def test_node_log_message(
        self, module: ModuleType, tmp_path: Path, node_level: str, kept: bool
    ):
        fpath_log = tmp_path / "logs.log"

        # same name and message format as in Fed-BioMed's Requests class
        def print_node_log_message() -> None:
            logging.getLogger("fedbiomed").info(
                f"\033[1m{node_level}\033[0m\n"
                "\t\t\t\t\t\033[1m NODE\033[0m NODE_1\n"
                "\t\t\t\t\t\033[1m MESSAGE:\033[0m node message\033[0m\n"
            )

        with module.log_to_file(fpath_log):
            print_node_log_message()
        text = fpath_log.read_text()
        assert ("node message" in text) is kept
        if kept:
            assert f"fedbiomed INFO     - {node_level}\n" in text
            assert "\033" not in text

    def test_info_not_from_node(self, module: ModuleType, tmp_path: Path):
        # the first line is only parsed for messages forwarded from the nodes
        fpath_log = tmp_path / "logs.log"
        with module.log_to_file(fpath_log):
            logging.getLogger("fedbiomed").info("WARNING\nnot from a node")
        assert fpath_log.read_text() == ""

    def test_handler_removed_on_error(self, module: ModuleType, tmp_path: Path):
        fbm_logger = logging.getLogger("fedbiomed")
        handlers_before = (list(module.logger.handlers), list(fbm_logger.handlers))
        with pytest.raises(RuntimeError), module.log_to_file(tmp_path / "logs.log"):
            raise RuntimeError
        assert (list(module.logger.handlers), list(fbm_logger.handlers)) == (
            handlers_before
        )


class TestPrintLoggerMessagesAtExit:
    def test_print_logger_messages_at_exit(
        self, module: ModuleType, capsys: pytest.CaptureFixture
    ):
        with module.print_logger_messages_at_exit():
            module.logger.info("[age] first")
            logging.getLogger("fedbiomed").warning("not printed")
            assert capsys.readouterr().out == ""
            module.logger.warning("[age] second")
        assert capsys.readouterr().out == (
            "INFO     - [age] first\nWARNING  - [age] second\n"
        )
        assert not any(
            isinstance(handler, module._ListHandler)
            for handler in module.logger.handlers
        )

    def test_printed_on_error(self, module: ModuleType, capsys: pytest.CaptureFixture):
        with pytest.raises(RuntimeError), module.print_logger_messages_at_exit():
            module.logger.info("[age] before error")
            raise RuntimeError
        assert capsys.readouterr().out == "INFO     - [age] before error\n"
