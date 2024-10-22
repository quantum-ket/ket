"""Wrapper for Libket C API."""

from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from ctypes import (
    CFUNCTYPE,
    Structure,
    c_uint32,
    c_void_p,
    c_size_t,
    POINTER,
    c_bool,
    c_uint8,
    c_int32,
    c_uint64,
    c_double,
)
from functools import reduce
from operator import iconcat
from typing import Literal
import weakref
from os import environ
from os.path import dirname
from .wrapper import load_lib, os_lib_name


HADAMARD = 0
PAULI_X = 1
PAULI_Y = 2
PAULI_Z = 3
ROTATION_X = 10
ROTATION_Y = 20
ROTATION_Z = 30
PHASE_SHIFT = 31

SUCCESS = 0
CONTROL_TWICE = 1
DATA_NOT_AVAILABLE = 2
DEALLOCATED_QUBIT = 3
QUBIT_INDEX_OUT_OF_BOUNDS = 4
NUMBER_OF_QUBITS_EXCEEDED = 5
NO_ADJ = 6
NO_CTRL = 7
NON_GATE_INSTRUCTION_IN_ADJ = 8
TARGET_IN_CONTROL = 9
PROCESS_READY_TO_EXECUTE = 10
UNEXPECTED_RESULT_DATA = 11
DUMP_NOT_ALLOWED = 12
EXP_VALUE_NOT_ALLOWED = 13
SAMPLE_NOT_ALLOWED = 14
MEASURE_NOT_ALLOWED = 15
UNDEFINED_ERROR = 16


class BatchCExecution(Structure):  # pylint: disable=too-few-public-methods
    """C BatchCExecution Structure"""

    _fields_ = [
        (
            "submit_execution",
            CFUNCTYPE(None, POINTER(c_uint8), c_size_t, POINTER(c_uint8), c_size_t),
        ),
        ("get_results", CFUNCTYPE(None, POINTER(POINTER(c_uint8)), POINTER(c_size_t))),
    ]


API_argtypes = {
    # 'ket_type_method': ([input_list], [output_list]),
    "ket_set_log_level": ([c_uint32], []),
    "ket_build_info": ([], [POINTER(c_uint8), c_size_t]),
    "ket_process_new": ([c_void_p], [c_void_p]),
    "ket_process_delete": ([c_void_p], []),
    "ket_process_allocate_qubit": ([c_void_p], [c_size_t]),
    "ket_process_apply_gate": (
        [c_void_p, c_int32, c_double, c_size_t],
        [],
    ),
    "ket_process_apply_global_phase": (
        [c_void_p, c_double],
        [],
    ),
    "ket_process_measure": ([c_void_p, POINTER(c_size_t), c_size_t], [c_size_t]),
    "ket_hamiltonian_new": ([], [c_void_p]),
    "ket_hamiltonian_add": (
        [c_void_p, POINTER(c_int32), c_size_t, POINTER(c_size_t), c_size_t, c_double],
        [],
    ),
    "ket_process_exp_value": (
        [c_void_p, c_void_p],
        [c_size_t],
    ),
    "ket_process_sample": (
        [c_void_p, POINTER(c_size_t), c_size_t, c_uint64],
        [c_size_t],
    ),
    "ket_process_dump": ([c_void_p, POINTER(c_size_t), c_size_t], [c_size_t]),
    "ket_process_ctrl_push": ([c_void_p, POINTER(c_size_t), c_size_t], []),
    "ket_process_ctrl_pop": ([c_void_p], []),
    "ket_process_ctrl_stack": ([c_void_p], []),
    "ket_process_ctrl_unstack": ([c_void_p], []),
    "ket_process_adj_begin": ([c_void_p], []),
    "ket_process_adj_end": ([c_void_p], []),
    "ket_process_execute": ([c_void_p], []),
    "ket_process_transpile": ([c_void_p], []),
    "ket_process_instructions_json": (
        [c_void_p, POINTER(c_uint8), c_size_t],
        [c_size_t],
    ),
    "ket_process_isa_instructions_json": (
        [c_void_p, POINTER(c_uint8), c_size_t],
        [c_size_t],
    ),
    "ket_process_metadata_json": ([c_void_p, POINTER(c_uint8), c_size_t], [c_size_t]),
    "ket_process_get_qubit_status": ([c_void_p, c_size_t], [c_bool, c_bool]),
    "ket_process_get_measurement": ([c_void_p, c_size_t], [c_bool, c_uint64]),
    "ket_process_get_exp_value": ([c_void_p, c_size_t], [c_bool, c_double]),
    "ket_process_get_sample": (
        [c_void_p, c_size_t],
        [c_bool, POINTER(c_uint64), POINTER(c_uint64), c_size_t],
    ),
    "ket_process_get_dump_size": ([c_void_p, c_size_t], [c_bool, c_size_t]),
    "ket_process_get_dump": (
        [c_void_p, c_size_t, c_size_t],
        [POINTER(c_uint64), c_size_t, c_double, c_double],
    ),
    "ket_make_configuration": (
        [
            c_size_t,
            POINTER(BatchCExecution),
            c_int32,
            c_int32,
            c_int32,
            c_int32,
            c_bool,
            POINTER(c_size_t),
            c_size_t,
            c_int32,
            c_int32,
        ],
        [c_void_p],
    ),
}


def libket_path():
    """Get Libket shared library path"""

    if "LIBKET_PATH" in environ:
        path = environ["LIBKET_PATH"]
    else:
        path = f'{dirname(__file__)}/libs/{os_lib_name("ket")}'

    return path


API = load_lib("Libket", libket_path(), API_argtypes, "ket_error_message")


def set_log(level: int):
    """Set Libket log level"""

    API["ket_set_log_level"](level)


class Process:
    """Libket process wrapper from C API"""

    def __init__(self, configuration):
        self._as_parameter_ = API["ket_process_new"](configuration)
        self._finalizer = weakref.finalize(
            self, API["ket_process_delete"], self._as_parameter_
        )

    def __getattr__(self, name: str):
        return lambda *args: API["ket_process_" + name](self, *args)

    def __repr__(self) -> str:
        return "<Libket 'process'>"


def make_configuration(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    num_qubits: int,
    batch_execution,
    measure: Literal["Disable", "Allowed", "ValidAfter"],
    sample: Literal["Disable", "Allowed", "ValidAfter"],
    exp_value: Literal["Disable", "Allowed", "ValidAfter"],
    dump: Literal["Disable", "Allowed", "ValidAfter"],
    define_qpu: bool,
    coupling_graph: list[tuple[int, int]] | None,
    u4_gate_type: Literal["CX", "CZ"],
    u2_gate_set: Literal["All", "ZYZ", "RzSx"],
) -> Process:
    """Make a Libket configuration"""

    def feature_status(value):
        if value == "Disable":
            return 0
        if value == "Allowed":
            return 1
        return 2

    coupling_graph_size = len(coupling_graph) if coupling_graph else 0
    if coupling_graph_size > 0:
        coupling_graph = reduce(iconcat, coupling_graph, [])
        coupling_graph = (c_size_t * len(coupling_graph))(*coupling_graph)
    else:
        coupling_graph = None

    return API["ket_make_configuration"](
        num_qubits,
        batch_execution,
        feature_status(measure),
        feature_status(sample),
        feature_status(exp_value),
        feature_status(dump),
        define_qpu,
        coupling_graph,
        coupling_graph_size,
        1 if u4_gate_type == "CZ" else 0,
        {"All": 0, "ZYZ": 1, "RzSx": 2}[u2_gate_set],
    )
