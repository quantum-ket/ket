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
import json
from operator import iconcat
from typing import Literal
import weakref
from os import environ
from os.path import dirname
from abc import ABC, abstractmethod

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
            CFUNCTYPE(
                None,
                POINTER(c_uint8),
                c_size_t,
                POINTER(c_uint8),
                c_size_t,
                POINTER(c_double),
                c_size_t,
            ),
        ),
        ("get_results", CFUNCTYPE(None, POINTER(POINTER(c_uint8)), POINTER(c_size_t))),
        ("clear", CFUNCTYPE(None)),
    ]


API_argtypes = {
    # 'ket_type_method': ([input_list], [output_list]),
    "ket_set_log_level": ([c_uint32], []),
    "ket_build_info": ([], [POINTER(c_uint8), c_size_t]),
    "ket_process_new": ([c_void_p], [c_void_p]),
    "ket_process_delete": ([c_void_p], []),
    "ket_process_allocate_qubit": ([c_void_p], [c_size_t]),
    "ket_process_apply_gate": (
        [
            c_void_p,  # process
            c_int32,  # gate
            c_double,  # angle
            c_bool,  # use_param
            c_size_t,  # param_index
            c_size_t,  # target
        ],
        [],
    ),
    "ket_process_set_parameter": ([c_void_p, c_double], [c_size_t]),
    "ket_process_get_gradient": ([c_void_p, c_size_t], [c_bool, c_double]),
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
            c_size_t,  # num_qubits
            POINTER(BatchCExecution),  # batch_execution
            c_int32,  # measure
            c_int32,  # sample
            c_int32,  # exp_value
            c_int32,  # dump
            c_int32,  # gradient
            c_bool,  # define_qpu
            POINTER(c_size_t),  # coupling_graph
            c_size_t,  # coupling_graph_size
            c_int32,  # u4_gate
            c_int32,  # u2_gates
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


class BatchExecution(ABC):
    """Base class for constructing batch target executions."""

    def __init__(self):
        @CFUNCTYPE(
            None,
            POINTER(c_uint8),
            c_size_t,
            POINTER(c_uint8),
            c_size_t,
            POINTER(c_double),
            c_size_t,
        )
        def submit_execution(  # pylint: disable=too-many-arguments,too-many-positional-arguments
            logical_circuit,
            logical_circuit_size,
            physical_circuit,
            physical_circuit_size,
            parameters,
            parameters_size,
        ):
            logical_circuit = json.loads(
                bytearray(logical_circuit[:logical_circuit_size])
            )
            physical_circuit = json.loads(
                bytearray(physical_circuit[:physical_circuit_size])
            )
            parameters = parameters[:parameters_size]
            self.submit_execution(logical_circuit, physical_circuit, parameters)

        self._result_json = None
        self._result_len = None

        @CFUNCTYPE(None, POINTER(POINTER(c_uint8)), POINTER(c_size_t))
        def get_result(result_ptr, size):
            result = self.get_result()
            self._result_json = json.dumps(result).encode("utf-8")
            self._result_len = len(self._result_json)
            self._result_json = (c_uint8 * self._result_len)(*self._result_json)
            result_ptr[0] = self._result_json
            size[0] = self._result_len

        @CFUNCTYPE(None)
        def clear():
            self.clear()

        self.c_struct = BatchCExecution(
            submit_execution,
            get_result,
            clear,
        )

    @abstractmethod
    def submit_execution(
        self,
        logical_circuit: dict,
        physical_circuit: dict | None,
        parameters: list[float],
    ):
        """Get the quantum circuit to execute."""

    @abstractmethod
    def get_result(self) -> dict:
        """Get the result of the quantum circuit execution."""

    @abstractmethod
    def clear(self):
        """Clear the data to start a new execution."""

    def configure(self, **kwargs):
        """Configure the batch execution."""
        return make_configuration(batch_execution=self.c_struct, **kwargs)

    @staticmethod
    def get_qubit_index(qubit):
        """Get the qubit index from the qubit object."""
        match qubit:
            case {"Main": {"index": index}}:
                ...
            case {"index": index}:
                ...
            case _:
                raise ValueError(f"Invalid qubit: {qubit}")
        return index

    @staticmethod
    def get_gate_and_angle(gate):
        """Get the gate and possible angle from the gate object."""
        if isinstance(gate, str):
            return gate, {}
        return list(gate.items())[0]

    def process_instructions(self, instructions: dict):
        """Parse the instructions to execute."""
        for instruction in instructions:
            match instruction:
                case {"Gate": {"control": control, "gate": gate, "target": target}}:
                    control = list(map(self.get_qubit_index, control))
                    target = self.get_qubit_index(target)
                    gate, param = self.get_gate_and_angle(gate)
                    match gate:
                        case "Hadamard":
                            self.hadamard(target, control, **param)
                        case "PauliX":
                            self.pauli_x(target, control, **param)
                        case "PauliY":
                            self.pauli_y(target, control, **param)
                        case "PauliZ":
                            self.pauli_z(target, control, **param)
                        case "RotationX":
                            self.rotation_x(target, control, **param)
                        case "RotationY":
                            self.rotation_y(target, control, **param)
                        case "RotationZ":
                            self.rotation_z(target, control, **param)
                        case "Phase":
                            self.phase(target, control, **param)
                case {"ExpValue": {"index": index, "hamiltonian": hamiltonian}}:
                    self.exp_value(index, hamiltonian)
                case {"Measure": {"index": index, "qubits": qubits}}:
                    qubits = list(map(self.get_qubit_index, qubits))
                    self.measure(index, qubits)
                case {"Sample": {"index": index, "qubits": qubits, "shots": shots}}:
                    qubits = list(map(self.get_qubit_index, qubits))
                    self.sample(index, qubits, shots)
                case {"Dump": {"index": index, "qubits": qubits}}:
                    qubits = list(map(self.get_qubit_index, qubits))
                    self.dump(index, qubits)
                case "Identity":
                    continue
                case _:
                    raise ValueError(f"Invalid instruction: {instruction}")

    def pauli_x(self, target, control):
        """Apply a Pauli-X gate to the target qubit."""
        raise NotImplementedError("Pauli-X gate not implemented")

    def pauli_y(self, target, control):
        """Apply a Pauli-Y gate to the target qubit."""
        raise NotImplementedError("Pauli-Y gate not implemented")

    def pauli_z(self, target, control):
        """Apply a Pauli-Z gate to the target qubit."""
        raise NotImplementedError("Pauli-Z gate not implemented")

    def hadamard(self, target, control):
        """Apply a Pauli-Z gate to the target qubit."""
        raise NotImplementedError("Hadamard gate not implemented")

    def rotation_x(self, target, control, **kwargs):
        """Apply a X-Rotation gate to the target qubit."""
        raise NotImplementedError("X-Rotation gate not implemented")

    def rotation_y(self, target, control, **kwargs):
        """Apply a Y-Rotation gate to the target qubit."""
        raise NotImplementedError("Y-Rotation gate not implemented")

    def rotation_z(self, target, control, **kwargs):
        """Apply a Z-Rotation gate to the target qubit."""
        raise NotImplementedError("Z-Rotation gate not implemented")

    def phase(self, target, control, **kwargs):
        """Apply a Phase gate to the target qubit."""
        raise NotImplementedError("Phase gate not implemented")

    def exp_value(self, index, hamiltonian):
        """Compute the expectation value."""
        raise NotImplementedError("Expectation value not implemented")

    def measure(self, index, qubits) -> int:
        """Measure the qubits."""
        raise NotImplementedError("Measurement not implemented")

    def sample(self, index, qubits, shots):
        """Sample the qubits."""
        raise NotImplementedError("Sampling not implemented")

    def dump(self, index, qubits):
        """Dump the qubits."""
        raise NotImplementedError("Dump not implemented")


_FEATURE_STATUS = {"Disable": 0, "Allowed": 1, "ValidAfter": 2}


def make_configuration(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    num_qubits: int,
    batch_execution,
    measure: Literal["Disable", "Allowed", "ValidAfter"],
    sample: Literal["Disable", "Allowed", "ValidAfter"],
    exp_value: Literal["Disable", "Allowed", "ValidAfter"],
    dump: Literal["Disable", "Allowed", "ValidAfter"],
    gradient: Literal["Disable", "ParameterShift", "SupportsGradient"],
    define_qpu: bool,
    coupling_graph: list[tuple[int, int]] | None,
    u4_gate_type: Literal["CX", "CZ"],
    u2_gate_set: Literal["All", "ZYZ", "RzSx"],
) -> Process:
    """Make a Libket configuration"""

    coupling_graph_size = len(coupling_graph) if coupling_graph else 0
    if coupling_graph_size > 0:
        coupling_graph = reduce(iconcat, coupling_graph, [])
        coupling_graph = (c_size_t * len(coupling_graph))(*coupling_graph)
    else:
        coupling_graph = None

    return API["ket_make_configuration"](
        num_qubits,  # num_qubits
        batch_execution,  # batch_execution
        _FEATURE_STATUS[measure],  # measure
        _FEATURE_STATUS[sample],  # sample
        _FEATURE_STATUS[exp_value],  # exp_value
        _FEATURE_STATUS[dump],  # dump
        {"Disable": 0, "ParameterShift": 1, "SupportsGradient": 2}[
            gradient
        ],  # gradient
        define_qpu,  # define_qpu
        coupling_graph,  # coupling_graph
        coupling_graph_size,  # coupling_graph_size
        1 if u4_gate_type == "CZ" else 0,  # u4_gate
        {"All": 0, "ZYZ": 1, "RzSx": 2}[u2_gate_set],  # u2_gates
    )
