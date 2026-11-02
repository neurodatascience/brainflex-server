"""Logging helpers for the experiment scripts."""

import logging
import re
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

LOG_FORMAT_FILE = "%(asctime)s %(name)s %(levelname)-8s - %(message)s"
LOG_FORMAT_SUMMARY = "%(levelname)-8s - %(message)s"
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")  # terminal formatting codes

logger = logging.getLogger("brainflex")
logger.setLevel(logging.INFO)
fbm_logger = logging.getLogger("fedbiomed")


class _ListHandler(logging.Handler):
    """Keep the formatted log messages in memory."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(self.format(record))


class _StripAnsiFormatter(logging.Formatter):
    """Remove the terminal formatting codes."""

    def format(self, record: logging.LogRecord) -> str:
        return _ANSI_ESCAPE.sub("", super().format(record))


def _get_node_log_level(record: logging.LogRecord) -> int | None:
    """Get the original level of a node log message forwarded to the researcher."""
    if record.funcName != "print_node_log_message":
        return None
    level_name = _ANSI_ESCAPE.sub("", record.getMessage().split("\n", 1)[0]).strip()
    level = logging.getLevelName(level_name)
    return level if isinstance(level, int) else None


def _filter_log_file_record(record: logging.LogRecord) -> bool:
    """Keep all script messages but only the Fed-BioMed warnings and errors."""
    if record.name == fbm_logger.name or record.name.startswith(f"{fbm_logger.name}."):
        level = _get_node_log_level(record) or record.levelno
        # security audit events are excluded from Fed-BioMed's console handler too
        return level >= logging.WARNING and not getattr(record, "is_security", False)
    return True


@contextmanager
def log_to_file(fpath_log: Path) -> Generator[None, None, None]:
    """Write the script messages and Fed-BioMed warnings/errors to a file."""
    fpath_log.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(fpath_log, mode="w", encoding="utf-8")
    handler.setFormatter(_StripAnsiFormatter(LOG_FORMAT_FILE))
    handler.addFilter(_filter_log_file_record)
    for logger_ in (logger, fbm_logger):
        logger_.addHandler(handler)
    try:
        yield
    finally:
        for logger_ in (logger, fbm_logger):
            logger_.removeHandler(handler)
        handler.close()


@contextmanager
def print_logger_messages_at_exit() -> Generator[None, None, None]:
    """Print the script messages."""
    handler = _ListHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT_SUMMARY))
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        print("\n".join(handler.messages))
