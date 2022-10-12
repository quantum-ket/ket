from __future__ import annotations
#  Copyright 2020, 2021 Evandro Chagas Ribeiro da Rosa <evandro.crr@posgrad.ufsc.br>
#  Copyright 2020, 2021 Rafael de Santiago <r.santiago@ufsc.br>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
from typing import Any
from ket.base import process_begin, process_end, process_top, process_last, exec_quantum
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

    process_last().serialize_metrics(JSON)
    return json.loads(from_u8_to_str(*process_last().get_serialized_metrics()[:-1]))


def quantum_code_last() -> list[dict] | None:
    """Returns the quantum code of the last executed process

    The return is a json serialization of a struct CodeBlock list from Libket.
    See https://gitlab.com/quantum-ket/libket/-/blob/main/src/code_block.rs for more details.
    """

    process_last().serialize_quantum_code(JSON)
    return json.loads(from_u8_to_str(*process_last().get_serialized_quantum_code()[:-1]))


def quantum_exec_time() -> float | None:
    """Returns the quantum execution time in seconds of the last executed process

    If no processes have been run, this function returns `None`.
    """
    if process_last() is not None:
        return process_last().exec_time().value
    return None


def quantum_exec_timeout(timeout: int):
    """Set the quantum execution timeout

    Args:
        timeout: timeout in seconds.
    """

    process_top().set_timeout(timeout)


class run:  # pylint: disable=invalid-name
    """Execute in a new process

    Run the quantum operations in a new separated process.

    :Usage:

    .. code-block:: ket

        with run():
            ...

    """

    def __enter__(self):
        process_begin()

    def __exit__(self, type, value, tb):  # pylint: disable=redefined-builtin, invalid-name
        process_end()
