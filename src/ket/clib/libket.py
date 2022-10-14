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

from ctypes import (c_void_p, c_size_t, POINTER, c_bool, c_uint8,
                    c_int32, c_uint64, c_int64, c_double, c_char)
import weakref
from os import environ
from os.path import dirname
from .wrapper import load_lib, os_lib_name

EQ = 0
NEQ = 1
GT = 2
GEQ = 3
LT = 4
LEQ = 5
ADD = 6
SUB = 7
MUL = 8
MOD = 9
DIV = 10
SLL = 11
SRL = 12
AND = 13
OR = 14
XOR = 15

PAULI_X = 0
PAULI_Y = 1
PAULI_Z = 2
HADAMARD = 3
PHASE = 4
RX = 5
RY = 6
RZ = 7

SUCCESS = 0
CONTROL_TWICE = 1
DATA_NOT_AVAILABLE = 2
DEALLOCATED_QUBIT = 3
FAIL_TO_PARSE_RESULT = 4
NO_ADJ = 5
NO_CTRL = 6
NON_GATE_INSTRUCTION = 7
NOT_BIN = 8
NOT_JSON = 9
NOT_UNITARY = 10
PLUGIN_ON_CTRL = 11
TARGET_ON_CONTROL = 12
TERMINATED_BLOCK = 13
UNDEFINED_CLASSICAL_OP = 14
UNDEFINED_DATA_TYPE = 15
UNDEFINED_GATE = 16
UNEXPECTED_RESULT_DATA = 17
UNMATCHED_PID = 18
UNDEFINED_ERROR = 19

JSON = 0
BIN = 1

API_argtypes = {
    # 'ket_type_method': ([input_list], [output_list]),
    'ket_process_new': ([c_size_t], [c_void_p]),
    'ket_process_delete': ([c_void_p], []),
    'ket_process_allocate_qubit': ([c_void_p, c_bool], [c_void_p]),
    'ket_process_free_qubit': ([c_void_p, c_void_p, c_bool], []),
    'ket_process_apply_gate': ([c_void_p, c_int32, c_double, c_void_p], []),
    'ket_process_apply_plugin': ([c_void_p, POINTER(c_char), POINTER(c_char), POINTER(c_void_p), c_size_t], []),  # pylint: disable=C0301
    'ket_process_measure': ([c_void_p, POINTER(c_void_p), c_size_t], [c_void_p]),
    'ket_process_ctrl_push': ([c_void_p, POINTER(c_void_p), c_size_t], []),
    'ket_process_ctrl_pop': ([c_void_p], []),
    'ket_process_adj_begin': ([c_void_p], []),
    'ket_process_adj_end': ([c_void_p], []),
    'ket_process_get_label': ([c_void_p], [c_void_p]),
    'ket_process_open_block': ([c_void_p, c_void_p], []),
    'ket_process_jump': ([c_void_p, c_void_p], []),
    'ket_process_branch': ([c_void_p, c_void_p, c_void_p, c_void_p], []),
    'ket_process_dump': ([c_void_p, POINTER(c_void_p), c_size_t], [c_void_p]),
    'ket_process_add_int_op': ([c_void_p, c_int32, c_void_p, c_void_p], [c_void_p]),
    'ket_process_int_new': ([c_void_p, c_int64], [c_void_p]),
    'ket_process_int_set': ([c_void_p, c_void_p, c_void_p], []),
    'ket_process_prepare_for_execution': ([c_void_p], []),
    'ket_process_exec_time': ([c_void_p], [c_double]),
    'ket_process_set_timeout': ([c_void_p, c_uint64], []),
    'ket_process_serialize_metrics': ([c_void_p, c_int32], []),
    'ket_process_serialize_quantum_code': ([c_void_p, c_int32], []),
    'ket_process_get_serialized_metrics': ([c_void_p], [POINTER(c_uint8), c_size_t, c_int32]),
    'ket_process_get_serialized_quantum_code': ([c_void_p], [POINTER(c_uint8), c_size_t, c_int32]),
    'ket_process_set_serialized_result': ([c_void_p, POINTER(c_uint8), c_size_t, c_int32], []),
    'ket_qubit_delete': ([c_void_p], []),
    'ket_qubit_index': ([c_void_p], [c_size_t]),
    'ket_qubit_pid': ([c_void_p], [c_size_t]),
    'ket_qubit_allocated': ([c_void_p], [c_bool]),
    'ket_qubit_measured': ([c_void_p], [c_bool]),
    'ket_dump_delete': ([c_void_p], []),
    'ket_dump_states_size': ([c_void_p], [c_size_t]),
    'ket_dump_state': ([c_void_p, c_size_t], [POINTER(c_uint64), c_size_t]),
    'ket_dump_amplitudes_real': ([c_void_p], [POINTER(c_double), c_size_t]),
    'ket_dump_amplitudes_imag': ([c_void_p], [POINTER(c_double), c_size_t]),
    'ket_dump_available': ([c_void_p], [c_bool]),
    'ket_future_delete': ([c_void_p], []),
    'ket_future_value': ([c_void_p], [c_int64]),
    'ket_future_index': ([c_void_p], [c_size_t]),
    'ket_future_pid': ([c_void_p], [c_size_t]),
    'ket_future_available': ([c_void_p], [c_bool]),
    'ket_label_delete': ([c_void_p], []),
    'ket_label_index': ([c_void_p], [c_size_t]),
    'ket_label_pid': ([c_void_p], [c_size_t]),
}


def libket_path():
    """Get Libket path"""

    if "LIBKET_PATH" in environ:
        path = environ["LIBKET_PATH"]
    else:
        path = f'{dirname(__file__)}/libs/{os_lib_name("ket")}'

    return path


API = load_lib('Libket', libket_path(), API_argtypes, 'ket_error_message')


class Process:
    """Libket process"""

    def __init__(self, pid: int):
        self.pid = pid
        self._as_parameter_ = API['ket_process_new'](pid)
        self._finalizer = weakref.finalize(
            self, API['ket_process_delete'], self._as_parameter_)

    def __getattr__(self, name: str):
        return lambda *args: API['ket_process_' + name](self, *args)

    def __repr__(self) -> str:
        return f"<Libket 'process' ({self.pid})>"


class LibketQubit:
    """Libket qubit type"""

    def __init__(self, addr: c_void_p):
        self._as_parameter_ = addr
        self._finalizer = weakref.finalize(
            self, API['ket_qubit_delete'], self._as_parameter_)

    def __getattr__(self, name: str):
        return lambda *args: API['ket_qubit_' + name](self, *args)

    def __repr__(self) -> str:
        return f"<Libket 'qubit' {self.pid().value, self.index().value}>"


class LibketDump:
    """Libket dump type"""

    def __init__(self, addr: c_void_p):
        self._as_parameter_ = addr
        self._finalizer = weakref.finalize(
            self, API['ket_dump_delete'], self._as_parameter_)

    def __getattr__(self, name: str):
        return lambda *args: API['ket_dump_' + name](self, *args)

    def __repr__(self) -> str:
        return f"<Libket 'dump' {self.pid().value, self.index().value}>"


class LibketFuture:
    """Libket future type"""

    def __init__(self, addr: c_void_p):
        self._as_parameter_ = addr
        self._finalizer = weakref.finalize(
            self, API['ket_future_delete'], self._as_parameter_)

    def __getattr__(self, name: str):
        return lambda *args: API['ket_future_' + name](self, *args)

    def __repr__(self) -> str:
        return f"<Libket 'future' {self.pid().value, self.index().value}>"


class LibketLabel:
    """Libket label type"""

    def __init__(self, addr: c_void_p):
        self._as_parameter_ = addr
        self._finalizer = weakref.finalize(
            self, API['ket_label_delete'], self._as_parameter_)

    def __getattr__(self, name: str):
        return lambda *args: API['ket_label_' + name](self, *args)

    def __repr__(self) -> str:
        return f"<Libket 'label' {self.pid().value, self.index().value}>"
