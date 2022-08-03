from __future__ import annotations
import json
from typing import Any
from ket.base import process_begin, process_end, process_top, _process_last, _exec_time, exec_quantum
from ket.clib.libket import JSON
from ket.clib.wrapper import from_u8_to_str

__all__ = ['run', 'quantum_metrics', 'quantum_metrics_last', 'quantum_code_last',
           'quantum_code', 'quantum_exec_time', 'quantum_exec_timeout', 'exec_quantum']


def quantum_metrics() -> dict[str, Any]:
    """Returns the metrics of the current process

    The return is a json serialization of a struct Metrics from Libket.
    See https://gitlab.com/quantum-ket/libket/-/blob/main/src/ir.rs for more details.
    """

    process_top().serialize_metrics(JSON)
    return json.loads(from_u8_to_str(*process_top().get_serialized_metrics()[:-1]))


def quantum_code() -> list[dict]:
    """Returns the quantum code of the current process

    The return is a json serialization of a struct CodeBlock list from Libket.
    See https://gitlab.com/quantum-ket/libket/-/blob/main/src/code_block.rs for more details.
    """

    process_top().serialize_quantum_code(JSON)
    return json.loads(from_u8_to_str(*process_top().get_serialized_quantum_code()[:-1]))


def quantum_metrics_last() -> dict[str, Any] | None:
    """Returns the metrics of the last executed process

    If no processes have been run, this function returns `None`.

    The return is a json serialization of a struct Metrics from Libket.
    See https://gitlab.com/quantum-ket/libket/-/blob/main/src/ir.rs for more details.
    """

    _process_last.serialize_metrics(JSON)
    return json.loads(from_u8_to_str(*_process_last.get_serialized_metrics()[:-1]))


def quantum_code_last() -> list[dict] | None:
    """Returns the quantum code of the last executed process

    The return is a json serialization of a struct CodeBlock list from Libket.
    See https://gitlab.com/quantum-ket/libket/-/blob/main/src/code_block.rs for more details.
    """

    _process_last.serialize_quantum_code(JSON)
    return json.loads(from_u8_to_str(*_process_last.get_serialized_quantum_code()[:-1]))


def quantum_exec_time() -> float | None:
    """Returns the quantum execution time in seconds of the last executed process

    If no processes have been run, this function returns `None`.
    """
    return _exec_time


def quantum_exec_timeout(timeout: int):
    """Set the quantum execution timeout

    Args:
        timeout: timeout in seconds.
    """

    process_top().set_timeout(timeout)


class run:
    """Execute in a new process

    Run the quantum operations in a new separated process.

    :Usage:

    .. code-block:: ket

        with run():
            ...

    """

    def __enter__(self):
        process_begin()

    def __exit__(self, type, value, tb):
        process_end()
