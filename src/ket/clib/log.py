"""Logging for Ket, Libket and KBW.

Ket logs through the standard :mod:`logging` module, under the ``ket`` logger:

- ``ket.libket.*``: records from Libket, the Rust runtime (for example
  ``ket.libket.process``).
- ``ket.kbw.*``: records from KBW, the simulator (for example
  ``ket.kbw.dense_gpu``).
- ``ket.*``: records from the Python package (for example ``ket.ibm``).

Libket and KBW only produce the records that the ``ket.libket`` and
``ket.kbw`` loggers would show, so logging costs nothing when disabled. Change
the level with :func:`set_log_level`, or with the ``KET_LOG`` environment
variable (``debug``, ``info``, ``warning``, ``error``, or ``off``). If you set
the level on the ``ket`` loggers yourself, call :func:`set_log_level` with no
argument, or create a new :class:`~ket.Process`, to pass it on to the Rust
libraries.
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import logging
import os
import threading
from ctypes import CFUNCTYPE, c_char_p, c_int32

__all__ = ["set_log_level", "get_logger"]

LOGGER = logging.getLogger("ket")

# Rust log levels (see `ket_log_set_level`): 0 off, 1 error, 2 warn,
# 3 info, 4 debug, 5 trace. Trace records go to Python's level 5.
TRACE = 5
_FROM_RUST = {1: logging.ERROR, 2: logging.WARNING, 3: logging.INFO, 4: logging.DEBUG}
_LEVEL_NAMES = {
    "off": logging.CRITICAL + 10,
    "none": logging.CRITICAL + 10,
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "trace": TRACE,
}

logging.addLevelName(TRACE, "TRACE")

LogCallback = CFUNCTYPE(None, c_int32, c_char_p, c_char_p)

# Loaded Rust libraries: (logger name, set_level C function).
_LIBS: list[tuple[str, object]] = []
# Keeps the C callbacks alive while the libraries may call them.
_CALLBACKS: list = []
_LOCK = threading.Lock()


def get_logger(name: str) -> logging.Logger:
    """Return the logger of a Ket module, under the ``ket`` logger."""
    if name == "ket" or name.startswith("ket."):
        return logging.getLogger(name)
    return logging.getLogger(f"ket.{name}")


def _to_rust(level: int) -> int:
    """Python logging level to the most verbose Rust level it shows."""
    if level <= TRACE:
        return 5
    if level <= logging.DEBUG:
        return 4
    if level <= logging.INFO:
        return 3
    if level <= logging.WARNING:
        return 2
    if level <= logging.ERROR:
        return 1
    return 0


def _parse_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    try:
        return _LEVEL_NAMES[level.strip().lower()]
    except KeyError:
        raise ValueError(
            f"unknown log level {level!r}; use one of: {', '.join(_LEVEL_NAMES)}"
        ) from None


def _sync_rust_levels():
    """Pass the effective level of each library's logger on to the library."""
    for name, set_level in _LIBS:
        set_level(_to_rust(logging.getLogger(name).getEffectiveLevel()))


def set_log_level(level: int | str | None = None):
    """Set the log level of Ket, Libket and KBW.

    Args:
        level: A :mod:`logging` level (``logging.DEBUG``) or its name
            (``"debug"``, ``"info"``, ``"warning"``, ``"error"``, ``"off"``;
            ``"trace"`` shows every operation of Libket and KBW). ``None``
            keeps the level of the ``ket`` loggers and only passes it on to
            Libket and KBW, for when you configure :mod:`logging` yourself.

    Records go to the handlers of the ``ket`` logger or of the root logger
    (see :func:`logging.basicConfig`). Without any handler, Python prints
    warnings and errors to ``stderr``.

    Example:

        .. code-block:: python

            import ket
            ket.set_log_level("debug")
    """
    if level is not None:
        LOGGER.setLevel(_parse_level(level))
    with _LOCK:
        _sync_rust_levels()


def _logger_name(prefix: str, target: str) -> str:
    """Python logger name for a Rust log target.

    ``ket::process`` from Libket gives ``ket.libket.process``; ``kbw::dense_gpu``
    from KBW gives ``ket.kbw.dense_gpu``; targets of other crates (such as
    ``wgpu_core``) go under the library that logged them.
    """
    parts = [part for part in target.split("::") if part]
    if parts and parts[0] in ("ket", "kbw"):
        parts = parts[1:]
    return ".".join([prefix, *parts])


def install(prefix: str, set_callback, set_level):
    """Send the log records of a Rust library to the logger ``prefix``.

    Args:
        prefix: Logger name, such as ``"ket.libket"``.
        set_callback: The library's ``*_log_set_callback`` C function.
        set_level: The library's ``*_log_set_level`` C function.
    """

    def callback(level, target, message):
        # An exception here would only be printed by ctypes, and a record
        # that cannot be shown must not break the simulation.
        try:
            py_level = _FROM_RUST.get(level, TRACE)
            target = target.decode("utf-8", "replace") if target else ""
            logger = logging.getLogger(_logger_name(prefix, target))
            if logger.isEnabledFor(py_level):
                message = message.decode("utf-8", "replace") if message else ""
                logger.log(py_level, "%s", message)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    c_callback = LogCallback(callback)
    with _LOCK:
        _CALLBACKS.append(c_callback)
        _LIBS.append((prefix, set_level))
        set_callback(c_callback)
        set_level(_to_rust(logging.getLogger(prefix).getEffectiveLevel()))

    # Rust threads must not call into Python once the interpreter shuts down.
    # `LogCallback()` is a NULL callback: records go to stderr from then on.
    atexit.register(set_callback, LogCallback())


def _configure_from_env():
    """Apply ``KET_LOG``: set the level and, if logging is not configured,
    print the records to ``stderr``."""
    env = os.environ.get("KET_LOG", "").strip()
    if not env:
        return
    try:
        LOGGER.setLevel(_parse_level(env))
    except ValueError:
        LOGGER.warning("ignoring KET_LOG=%r: unknown log level", env)
        return
    if not LOGGER.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(levelname)s %(name)s] %(message)s")
        )
        LOGGER.addHandler(handler)


_configure_from_env()
