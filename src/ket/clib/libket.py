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
import json
from typing import Any, Literal
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
    "ket_process_instructions_json": (
        [c_void_p, POINTER(c_uint8), c_size_t],
        [c_size_t],
    ),
    "ket_process_isa_instructions_json": (
        [c_void_p, POINTER(c_uint8), c_size_t],
        [c_size_t],
    ),
    "ket_process_metadata_json": ([c_void_p, POINTER(c_uint8), c_size_t], [c_size_t]),
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
            POINTER(c_uint8),  # json
            c_size_t,  # json_size
            POINTER(BatchCExecution),  # batch_execution
        ],
        [c_void_p],
    ),
    "ket_process_save_sim_state": ([c_void_p, POINTER(c_uint8), c_size_t], [c_size_t]),
    "ket_process_load_sim_state": ([c_void_p, POINTER(c_uint8), c_size_t], []),
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
            POINTER(c_double),
            c_size_t,
        )
        def submit_execution(  # pylint: disable=too-many-arguments,too-many-positional-arguments
            circuit,
            circuit_size,
            parameters,
            parameters_size,
        ):
            circuit = json.loads(bytearray(circuit[:circuit_size]))

            parameters = parameters[:parameters_size]
            self.submit_execution(circuit, parameters)

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
        circuit: dict,
        parameters: list[float],
    ):
        """Get the quantum circuit to execute."""

    @abstractmethod
    def get_result(self) -> dict:
        """Get the result of the quantum circuit execution."""

    @abstractmethod
    def clear(self):
        """Clear the data to start a new execution."""

    @abstractmethod
    def connect(self):
        """Call configure with the appropriated arguments to generate the object."""

    def configure(self, **kwargs):
        """Configure the batch execution."""
        return make_configuration(batch_execution=self.c_struct, **kwargs)

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
                    self.measure(index, qubits)
                case {"Sample": {"index": index, "qubits": qubits, "shots": shots}}:
                    self.sample(index, qubits, shots)
                case {"Dump": {"index": index, "qubits": qubits}}:
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


_BASE_QPU = {
    "coupling_graph": None,
    "u2_gates": "All",  # All, ZYZ, RzSx
    "u4_gate": "CX",  # CX, CZ
}

# Unsupported, Basic, Advanced
_MANAGED_BY_TARGET = {
    "measure": "Unsupported",
    "sample": "Unsupported",
    "exp_value": "Unsupported",
    "dump": "Unsupported",
}

_CLASSICAL_SHADOWS = {"bias": (1, 1, 1), "samples": 1_000, "shots": 2048}


def make_configuration(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    num_qubits: int,
    batch_execution,
    execution_managed_by_target: dict[str, str] | None = None,
    direct_sample_exp_value: int | None = None,
    classical_shadows_exp_value: dict[str, Any] | None = None,
    qpu: dict[str, str] | None = None,
    gradient: Literal["ParameterShift", "NativeSupport"] | None = None,
) -> Process:
    """Make a Libket configuration"""

    if qpu is not None:
        qpu = {**_BASE_QPU, **qpu}

    # Ensure only one of the three is defined and at least one is provided
    defined_options = [
        execution_managed_by_target is not None,
        direct_sample_exp_value is not None,
        classical_shadows_exp_value is not None,
    ]
    if sum(defined_options) != 1:
        raise ValueError(
            "Exactly one of 'execution_managed_by_target', 'direct_sample_exp_value', "
            "or 'classical_shadows_exp_value' must be defined."
        )

    if execution_managed_by_target is not None:
        execution_protocol = {
            "ManagedByTarget": {**_MANAGED_BY_TARGET, **execution_managed_by_target}
        }
    elif direct_sample_exp_value is not None:
        execution_protocol = {"SampleBased": {"DirectSample": direct_sample_exp_value}}
    else:
        execution_protocol = {
            "SampleBased": {
                "ClassicalShadows": {
                    **_CLASSICAL_SHADOWS,
                    **classical_shadows_exp_value,
                }
            }
        }

    execution_target = {
        "num_qubits": num_qubits,
        "qpu": qpu,
        "execution_protocol": execution_protocol,
        "gradient": gradient,
    }

    execution_target_json = json.dumps(execution_target).encode("utf-8")
    execution_target_len = len(execution_target_json)
    execution_target_json = (c_uint8 * execution_target_len)(*execution_target_json)

    return API["ket_make_configuration"](
        execution_target_json,
        execution_target_len,
        batch_execution,
    )
