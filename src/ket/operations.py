"""Basic Ket type definitions."""
from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0


from contextlib import contextmanager
from ctypes import c_size_t


from .base import (
    Quant,
    Measurement,
    Samples,
    QuantumState,
    Pauli,
    Hamiltonian,
    ExpValue,
)

__all__ = [
    "control",
    "ctrl",
    "inverse",
    "adj",
    "around",
    "cat",
    "kron",
    "measure",
    "sample",
    "dump",
    "exp_value",
]


@contextmanager
def control(qubits: Quant):
    """Controlled scope."""
    process = qubits.process
    process.ctrl_push(
        (c_size_t * len(qubits.qubits))(*qubits.qubits),
        len(qubits.qubits),
    )
    try:
        yield
    finally:
        process.ctrl_pop()


def ctrl(control_qubits: Quant, gate):
    """Add control qubits to a gate call."""

    def inner(*args, **kwargs):
        with control(control_qubits):
            return control_qubits, gate(*args, **kwargs)

    return inner


def adj(gate):
    """Return the inverse of a gate."""

    def inner(*args, ket_process=None, **kwargs):
        process = None
        for arg in args:
            if hasattr(arg, "_get_ket_process"):
                arg_process = arg._get_ket_process()  # pylint: disable=protected-access
                if process is not None and process is not arg_process:
                    raise ValueError("parameter with different Ket processes")
                process = arg_process
        for arg in kwargs.values():
            if hasattr(arg, "_get_ket_process"):
                arg_process = arg._get_ket_process()  # pylint: disable=protected-access
                if process is not None and process is not arg_process:
                    raise ValueError("parameter with different Ket processes")
                process = arg_process

        if process is None:
            process = ket_process

        if process is None:
            raise ValueError("no Ket process found in parameters")

        with inverse(process):
            gate(*args, **kwargs)

    return inner


@contextmanager
def inverse(process):
    """Inverse scope."""
    process.adj_begin()
    try:
        yield
    finally:
        process.adj_end()


def cat(*gates):
    """Concatenate gates."""

    def inner(*args):
        for gate in gates:
            args = gate(*args)
            if not hasattr(args, "__iter__"):
                args = tuple(args)

        if len(args) == 1:
            return args[0]
        return args

    return inner


def kron(*gates):
    """Return the tensor product on the gates."""

    def inner(*args):
        return tuple(gate(arg) for gate, arg in zip(gates, args))

    return inner


@contextmanager
def around(gate, *arg, ket_process=None, **kwargs):
    """Apply a gate around a context."""
    gate(*arg, **kwargs)
    try:
        yield
    finally:
        adj(gate)(*arg, ket_process=ket_process, **kwargs)


def measure(qubits: Quant) -> Measurement:
    """Measure the qubits and return a :class:`~ket.base.measurement` object."""
    return Measurement(qubits)


def dump(qubits: Quant) -> QuantumState:
    """Get the quantum state"""
    return QuantumState(qubits)


def sample(qubits: Quant, shots: int = 2048) -> Samples:
    """Get the quantum state measurement samples"""
    return Samples(qubits, shots)


def exp_value(hamiltonian: Hamiltonian | Pauli) -> ExpValue:
    """Expected value for a quantum state."""
    return ExpValue(hamiltonian)
