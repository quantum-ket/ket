"""Wrapper for Libket C API."""

from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from ctypes import (
    CFUNCTYPE,
    c_char_p,
    c_int32,
    c_size_t,
    POINTER,
    c_uint64,
    c_double,
)
import json
from abc import ABC, abstractmethod

from . import (
    BatchCExecution,
    LiveCExecution,
    CNativeGateSet,
    API,
)


class BatchExecution(ABC):  # pylint: disable=too-many-instance-attributes
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

        @CFUNCTYPE(
            c_int32,
            c_char_p,  # gates_json
            POINTER(c_size_t),  # qubits_to_sample
            c_size_t,  # qubits_to_sample_len
            c_size_t,  # shots
            POINTER(c_char_p),  # sample_json
        )
        def sample_native(
            gates_json, qubits_to_sample, qubits_to_sample_len, shots, result_json
        ):
            gates = json.loads(gates_json)
            qubits = [qubits_to_sample[i] for i in range(qubits_to_sample_len)]
            sample_data, counts = self.sample_native(gates, qubits, shots)
            self._sample_native_result = json.dumps([sample_data, counts]).encode(
                "utf-8"
            )
            result_json[0] = self._sample_native_result
            return 0

        self._sample_native_result = None

        @CFUNCTYPE(
            c_int32,
            c_char_p,  # gates_json
            c_char_p,  # hamiltonian_list_json
            POINTER(c_double),  # result
        )
        def exp_value_native(
            gates_json,
            hamiltonian_list_json,
            result,
        ):
            gates = json.loads(gates_json)
            hamiltonian_list = json.loads(hamiltonian_list_json)
            values = self.exp_value_native(gates, hamiltonian_list)
            for i, v in enumerate(values):
                result[i] = v
            return 0

        self._grad_result = None

        @CFUNCTYPE(
            c_int32,
            c_char_p,  # gates_json
            c_char_p,  # hamiltonian_json
            POINTER(c_double),  # exp_result
            POINTER(c_char_p),  # grad
        )
        def gradient(  # pylint: disable=too-many-arguments,too-many-positional-arguments
            gates_json,
            hamiltonian_json,
            exp_result,
            grad_json,
        ):
            gates = json.loads(gates_json)
            hamiltonian = json.loads(hamiltonian_json)
            exp_val, grad_vals = self.gradient(gates, hamiltonian)
            exp_result[0] = exp_val
            self._grad_result = json.dumps(grad_vals).encode("utf-8")
            grad_json[0] = self._grad_result
            return 0

        self._cb_sample = sample
        self._cb_exp_value = exp_value
        self._cb_sample_native = sample_native
        self._cb_exp_value_native = exp_value_native
        self._cb_gradient = gradient

        self.c_struct = BatchCExecution(
            self._cb_sample,
            self._cb_exp_value,
            self._cb_sample_native,
            self._cb_exp_value_native,
            self._cb_gradient,
        )

    def configure(
        self,
        num_qubits: int,
        gradient: bool = False,
        coupling_graph: list[tuple[int, int]] | None = None,
        native_gate_set: NativeGateSet | None = None,
        decompose: bool = False,
    ):
        """Configure the batch execution and return a QuantumExecution pointer."""
        return make_batch_configuration(
            num_qubits,
            execution=self,
            gradient=gradient,
            coupling_graph=coupling_graph,
            native_gate_set=native_gate_set,
            decompose=decompose,
        )

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

    def sample_native(self, gates, qubits, shots):
        """Execute the circuit with native gates and compute expectation values.

        .. warning::
            This method is called by Libket and should not be called directly.
        """
        raise NotImplementedError("`sample_native` method not implemented")

    def exp_value_native(self, gates, hamiltonians):
        """Execute the circuit with native and compute expectation values.

        .. warning::
            This method is called by Libket and should not be called directly.
        """
        raise NotImplementedError("`exp_value_native` method not implemented")

    def gradient(self, gates, hamiltonian):
        """Execute the circuit, compute expectation values and the parameters gradient.

        .. warning::
            This method is called by Libket and should not be called directly.
        """
        raise NotImplementedError("`gradient` method not implemented")


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
            except Exception:  # pylint: disable=broad-exception-caught
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

        @CFUNCTYPE(
            c_int32,
            c_char_p,  # json
        )
        def compute_native_gates(json_ptr):
            gate_dict = json.loads(json_ptr)
            return self.compute_native_gates(gate_dict)

        self._cb_compute_gate = compute_gate
        self._cb_compute_native_gates = compute_native_gates
        self._cb_measure = measure
        self._cb_dump = dump
        self._cb_sample = sample
        self._cb_exp_value = exp_value

        self.c_struct = LiveCExecution(
            self._cb_compute_gate,
            self._cb_compute_native_gates,
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

    def compute_native_gates(self, gates) -> dict:
        """Compute native gates."""
        raise NotImplementedError("`compute_native_gates` method not implemented")

    @abstractmethod
    def connect(self):
        """Call configure with the appropriated arguments to generate the object."""

    def configure(self, num_qubits: int, decompose: bool = False):
        """Configure the live execution and return a QuantumExecution pointer."""
        return make_live_configuration(num_qubits, execution=self, decompose=decompose)


class NativeGateSet(ABC):
    """Base class for constructing a Python-friendly native gate set.

    Subclass this to define how abstract gates are translated into the
    instruction set of a specific backend (hardware or simulator).  The
    three methods, :meth:`translate`, :meth:`cnot`, and :meth:`swap`,
    work at the Python level; all JSON marshalling with the C layer is
    handled internally.

    A :class:`NativeGateSet` instance can be passed directly to
    :meth:`BatchExecution.configure`.

    Example::

        class MyGateSet(NativeGateSet):
            def translate(self, matrix, target):
                # matrix is [[u00, u01], [u10, u11]] (complex values)
                # Return a list of native gate dicts
                return [{"U": {"matrix": str(matrix), "target": target}}]

            def cnot(self, control, target):
                return [{"CNOT": {"control": control, "target": target}}]

            def swap(self, a, b):
                return [{"SWAP": {"a": a, "b": b}}]

        gate_set = MyGateSet()
        ptr = batch_exec.configure(native_gate_set=gate_set)
    """

    def __init__(self):
        @CFUNCTYPE(
            c_int32,
            c_char_p,  # gate_json
            c_size_t,  # target
            POINTER(c_char_p),  # native_gate_json
        )
        def translate(gate_json, target, native_gate_json):
            flat = json.loads(gate_json)
            matrix = [
                [complex(row[i], row[i + 1]) for i in range(0, len(row), 2)]
                for row in flat
            ]
            result = self.translate(matrix, target)
            self._translate_result = json.dumps(result).encode("utf-8")
            native_gate_json[0] = self._translate_result
            return 0

        self._translate_result = None

        @CFUNCTYPE(
            c_int32,
            c_size_t,  # control
            c_size_t,  # target
            POINTER(c_char_p),  # native_gate_json  (out)
        )
        def cnot(control, target, native_gate_json):
            result = self.cnot(control, target)
            self._cnot_result = json.dumps(result).encode("utf-8")
            native_gate_json[0] = self._cnot_result
            return 0

        self._cnot_result = None

        # Keep references alive for the duration of the object's lifetime.
        self._cb_translate = translate
        self._cb_cnot = cnot

        self.c_struct = CNativeGateSet(
            self._cb_translate,
            self._cb_cnot,
        )

    @abstractmethod
    def translate(self, matrix: list[list[complex]], target: int) -> list:
        """Translate a single-qubit gate into native gate instructions.

        The gate is described by its 2x2 unitary matrix as a list of two
        rows of complex amplitudes::

            [[u00, u01],
             [u10, u11]]

        Args:
            matrix: The 2x2 unitary matrix as complex values.
            target: Index of the target qubit.

        Returns:
            A list of native gate objects (serialisable to JSON) that
            implement the given unitary on ``target``.
        """

    @abstractmethod
    def cnot(self, control: int, target: int) -> list:
        """Produce native instructions implementing a CNOT gate.

        Args:
            control: Index of the control qubit.
            target: Index of the target qubit.

        Returns:
            A list of native gate objects that implement CNOT.
        """


def make_live_configuration(
    num_qubits: int,
    execution: LiveExecution,
    native_gate_set: NativeGateSet | None = None,
    decompose: bool = False,
):
    """
    Constructs a LiveConfiguration.

    Args:
        execution (LiveExecution): The execution wrapper instance.
    """

    if native_gate_set is not None:
        native_gate_set_ptr = native_gate_set.c_struct
    else:
        native_gate_set_ptr = None

    return API["ket_quantum_execution_live"](
        num_qubits,
        execution.c_struct,
        decompose,
        native_gate_set_ptr,
    )


def make_batch_configuration(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    num_qubits: int,
    execution: BatchExecution,
    gradient: bool = False,
    coupling_graph: list[tuple[int, int]] | None = None,
    exp_value_strategy=None,
    native_gate_set: NativeGateSet | None = None,
    decompose: bool = False,
):
    """
    Constructs a BatchConfiguration.

    Args:
        execution (BatchExecution): The execution wrapper instance.
    """

    coupling_graph_json = json.dumps(coupling_graph).encode("utf-8")

    if exp_value_strategy is not None:
        exp_value_strategy_json = json.dumps(exp_value_strategy).encode("utf-8")
    else:
        exp_value_strategy_json = json.dumps("Native").encode("utf-8")

    if native_gate_set is not None:
        native_gate_set_ptr = native_gate_set.c_struct
    else:
        native_gate_set_ptr = None

    return API["ket_quantum_execution_batch"](
        num_qubits,
        execution.c_struct,
        native_gate_set_ptr,
        decompose,
        gradient,
        coupling_graph_json,
        exp_value_strategy_json,
    )
