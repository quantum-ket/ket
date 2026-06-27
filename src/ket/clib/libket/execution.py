"""Wrapper for Libket C API."""

from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from ctypes import (
    CFUNCTYPE,
    c_bool,
    c_char_p,
    c_int32,
    c_size_t,
    POINTER,
    c_uint64,
    c_double,
)
import json
from abc import ABC, abstractmethod

from . import BatchCExecution, LiveCExecution, CNativeGateSet, API


class BatchExecution(ABC):
    """Base class for constructing batch target executions."""

    def __init__(self):
        @CFUNCTYPE(
            c_int32,
            c_char_p,  # gates_json
            POINTER(c_size_t),  # qubits_to_sample
            c_size_t,  # qubits_to_sample_len
            c_size_t,  # shots
            POINTER(c_char_p),  # sample_json
        )
        def sample(  # pylint: disable=too-many-arguments,too-many-positional-arguments
            gates_json,
            qubits_to_sample,
            qubits_to_sample_len,
            shots,
            sample_json,
        ):
            gates = json.loads(gates_json)
            qubits_to_sample = qubits_to_sample[:qubits_to_sample_len]
            result = self.sample(gates, qubits_to_sample, shots)
            self._sample_result = json.dumps(result).encode("utf-8")
            sample_json[0] = self._sample_result
            return 0

        self._sample_result = None

        @CFUNCTYPE(
            c_int32,
            c_char_p,  # gates_json
            c_char_p,  # hamiltonian_list_json
            POINTER(c_double),  # result
        )
        def exp_value(  # pylint: disable=too-many-arguments,too-many-positional-arguments
            gates_json,
            hamiltonian_list_json,
            result,
        ):
            gates = json.loads(gates_json)
            hamiltonian_list = json.loads(hamiltonian_list_json)
            values = self.exp_value(gates, hamiltonian_list)
            for i, v in enumerate(values):
                result[i] = v
            return 0

        @CFUNCTYPE(c_bool)
        def exp_value_available():
            return True

        self._cb_sample = sample
        self._cb_exp_value = exp_value
        self._exp_value_available = exp_value_available

        self.c_struct = BatchCExecution(
            self._cb_sample,
            self._cb_exp_value,
            self._exp_value_available,
        )

    def configure(
        self,
        gradient: bool = False,
        coupling_graph: list[tuple[int, int]] | None = None,
        native_gate_set: CNativeGateSet | None = None,
    ):
        """Configure the batch execution and return a QuantumExecution pointer."""
        return make_batch_configuration(
            execution=self,
            gradient=gradient,
            coupling_graph=coupling_graph,
            native_gate_set=native_gate_set,
        )

    @staticmethod
    def get_gate_and_angle(gate):
        """Get the gate and possible angle from the gate object."""
        if isinstance(gate, str):
            return gate, {}
        return list(gate.items())[0]

    def sample(self, gates, qubits_to_sample, shots):
        """Execute the circuit and sample the given qubits.

        .. warning::
            This method is called by Libket and should not be called directly.
        """
        raise NotImplementedError("`sample` method not implemented")

    def exp_value(self, gates, hamiltonian_list):
        """Execute the circuit and compute expectation values.

        .. warning::
            This method is called by Libket and should not be called directly.
        """
        raise NotImplementedError("`exp_value` method not implemented")


class LiveExecution(ABC):  # pylint: disable=too-many-instance-attributes
    """Base class for constructing live target executions."""

    def __init__(self):  # pylint: disable=too-many-statements
        @CFUNCTYPE(
            c_int32,
            c_char_p,  # gate_instruction_json
        )
        def compute_gate(gate_json):
            gate_dict = json.loads(gate_json)
            gate_name = gate_dict["gate"]
            target = gate_dict["target"]
            control = sorted(gate_dict.get("control", []))
            value = None

            if isinstance(gate_name, dict):
                gate_name, value_dict = list(gate_name.items())[0]
                value = value_dict.get("Value")

            try:
                match gate_name:
                    case "Hadamard":
                        self.hadamard(target, control)
                    case "PauliX":
                        self.pauli_x(target, control)
                    case "PauliY":
                        self.pauli_y(target, control)
                    case "PauliZ":
                        self.pauli_z(target, control)
                    case "RotationX":
                        self.rotation_x(target, control, value)
                    case "RotationY":
                        self.rotation_y(target, control, value)
                    case "RotationZ":
                        self.rotation_z(target, control, value)
                    case "Phase":
                        self.phase(target, control, value)
            except:
                return 13
            return 0

        @CFUNCTYPE(c_int32, POINTER(c_size_t), c_size_t, POINTER(c_uint64))
        def measure(qubits, qubits_size, result):
            qubits = qubits[:qubits_size]
            result[0] = self.measure(qubits)
            return 0

        self._dump_json = None

        @CFUNCTYPE(
            c_int32,
            POINTER(c_size_t),
            c_size_t,
            POINTER(c_char_p),
        )
        def dump(qubits, qubits_size, result_json):
            qubits = qubits[:qubits_size]
            result = self.dump(qubits)
            self._dump_json = json.dumps(result).encode("utf-8")
            result_json[0] = self._dump_json
            return 0

        self._sample_json = None

        @CFUNCTYPE(
            c_int32,
            POINTER(c_size_t),
            c_size_t,
            c_size_t,
            POINTER(c_char_p),
        )
        def sample(qubits, qubits_size, shots, result_json):
            qubits = qubits[:qubits_size]
            result = self.sample(qubits, shots)
            self._sample_json = json.dumps(result).encode("utf-8")
            result_json[0] = self._sample_json
            return 0

        @CFUNCTYPE(c_int32, c_char_p, POINTER(c_double))
        def exp_value(h_json, result):
            h = json.loads(h_json)
            result[0] = self.exp_value(h)
            return 0

        self._cb_compute_gate = compute_gate
        self._cb_measure = measure
        self._cb_dump = dump
        self._cb_sample = sample
        self._cb_exp_value = exp_value

        self.c_struct = LiveCExecution(
            self._cb_compute_gate,
            self._cb_measure,
            self._cb_dump,
            self._cb_sample,
            self._cb_exp_value,
        )

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
        """Apply a Hadamard gate to the target qubit."""
        raise NotImplementedError("Hadamard gate not implemented")

    def rotation_x(self, target, control, angle):
        """Apply a X-Rotation gate to the target qubit."""
        raise NotImplementedError("X-Rotation gate not implemented")

    def rotation_y(self, target, control, angle):
        """Apply a Y-Rotation gate to the target qubit."""
        raise NotImplementedError("Y-Rotation gate not implemented")

    def rotation_z(self, target, control, angle):
        """Apply a Z-Rotation gate to the target qubit."""
        raise NotImplementedError("Z-Rotation gate not implemented")

    def phase(self, target, control, angle):
        """Apply a Phase gate to the target qubit."""
        raise NotImplementedError("Phase gate not implemented")

    def measure(self, qubits: list[int]) -> int:
        """Measure the qubits and return the result as an integer."""
        raise NotImplementedError("`measure` method not implemented")

    def exp_value(self, hamiltonian: dict):
        """Compute the expectation value of the given Hamiltonian."""
        raise NotImplementedError("`exp_value` method not implemented")

    def sample(self, qubits: list[int], shots: int):
        """Sample the state of the qubits."""
        raise NotImplementedError("`sample` method not implemented")

    def dump(self, qubits: list[int]) -> dict:
        """Dump the state of the qubits."""
        raise NotImplementedError("`dump` method not implemented")

    @abstractmethod
    def connect(self):
        """Call configure with the appropriated arguments to generate the object."""

    def configure(self):
        """Configure the live execution and return a QuantumExecution pointer."""
        return make_live_configuration(execution=self)


def make_live_configuration(execution: LiveExecution, decompose: bool = False):
    """Create a live QuantumExecution from a LiveExecution instance."""
    return API["ket_quantum_execution_live"](execution.c_struct, decompose)


def make_batch_configuration(
    execution: BatchExecution,
    gradient: bool = False,
    coupling_graph: list[tuple[int, int]] | None = None,
    native_gate_set: CNativeGateSet | None = None,
):
    """Create a batch QuantumExecution from a BatchExecution instance."""
    coupling_graph_json = json.dumps(coupling_graph).encode("utf-8")

    native_gate_set_ptr = native_gate_set if native_gate_set is not None else None

    return API["ket_quantum_execution_batch"](
        execution.c_struct,
        native_gate_set_ptr,
        gradient,
        coupling_graph_json,
    )
