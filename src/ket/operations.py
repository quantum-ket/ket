"""Functions to manipulate quantum operations.

This module provides essential functions in the Ket library for manipulating quantum operations. 
It includes functions for controlled and inverse operations, facilitating quantum circuit
construction.
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0


from contextlib import contextmanager
from ctypes import c_size_t
from functools import reduce
from operator import add
from typing import Any, Callable, Sequence


from .base import (
    Process,
    Quant,
    Measurement,
    Samples,
)
from .quantumstate import QuantumState


from .expv import (
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
def control(control_qubits: Quant):
    r"""Controlled scope.

    Opens a controlled scope where quantum operations are applied only if the qubits of the
    parameter ``control_qubits`` are in the state :math:`\ket{1}`.

    :Usage:

    .. code-block:: python

        with control(control_qubits):
            # Apply operations only if control_qubits are in the state |1>
            ...

    Example:

        .. code-block:: python

            from ket import *

            p = Process()

            c = p.alloc(2)
            a, b = q.alloc(2)

            # CNOT c[0] a
            with control(c[0]):
                X(a)

            # Toffoli c[0] c[1] a
            with control(c):
                X(a)

            # CSWAP c[0] a b
            with control(c[0]):
                SWAP(a, b)

    Args:
        control_qubits: The qubits to control the quantum operations.
    """
    if not isinstance(control_qubits, Quant):
        control_qubits = reduce(add, control_qubits)

    process = control_qubits.process
    process.ctrl_push(
        (c_size_t * len(control_qubits.qubits))(*control_qubits.qubits),
        len(control_qubits.qubits),
    )
    try:
        yield
    finally:
        process.ctrl_pop()


def ctrl(control_qubits: Quant, gate: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Add control qubits to a gate.

    Create a new callable that applies the given ``gate`` with the control qubits.

    :Usage:

    .. code-block:: python

        from ket import *

        p = Process()

        c = p.alloc(2)
        a, b = q.alloc(2)

        # CNOT c[0] a
        ctrl(c[0], X)(a)

        # Toffoli c[0] c[1] a
        ctrl(c, X)(a)

        # CSWAP c[0] a b
        ctrl(c[0], SWAP)(a, b)

    Args:
        control_qubits: The qubits to control the quantum operations.
        gate: The gate to apply with the control qubits.

    Returns:
        A new callable that applies the given gate with the control qubits.
    """

    def inner(*args, **kwargs):
        with control(control_qubits):
            return control_qubits, gate(*args, **kwargs)

    return inner


def _search_process(ket_process, args, kwargs):
    def inner(ket_process, arg):
        if hasattr(arg, "_get_ket_process"):
            arg_process = arg._get_ket_process()  # pylint: disable=protected-access
            if ket_process is not None and ket_process is not arg_process:
                raise ValueError("parameter with different Ket processes")
            ket_process = arg_process
        return ket_process

    def search(ket_process, args):
        for arg in args:
            if isinstance(arg, Sequence) and not isinstance(arg, str):
                for subarg in arg:
                    ket_process = inner(ket_process, subarg)
            else:
                ket_process = inner(ket_process, arg)
        return ket_process

    if ket_process is None:
        ket_process = search(ket_process, args)
        ket_process = search(ket_process, kwargs.values())

    if ket_process is None:
        raise ValueError("Ket process not found in the parameters")

    return ket_process


def adj(gate: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Return the inverse of a gate.

    Create a new callable that applies the inverse of the given ``gate``.

    The resulting callable will iterate over the parameters to identify the
    :class:`~ket.base.Process` that the inverse operation will be applied. If non of the parameters
    are a :class:`~ket.base.Quant`, the keyword argument ``ket_process`` must be given.

    :Usage:

    .. code-block:: python

        from ket import *

        p = Process()

        a, b = q.alloc(2)

        bell = cat(kron(H, I), CNOT)

        adj(bell)(a, b)

    Args:
        gate: The gate to apply the inverse.

    Returns:
        A new callable that applies the inverse of the given gate.

    """

    def inner(*args, ket_process: Process | None = None, **kwargs) -> Any:
        ket_process = _search_process(ket_process, args, kwargs)

        with inverse(ket_process):
            return gate(*args, **kwargs)

    return inner


@contextmanager
def inverse(process: Process):
    """Inverse scope.

    Open a scope where all the quantum operations are executed backwards.

    :Usage:

    .. code-block:: python

       with inverse(process):
            # Classical and quantum operations
            ...

    Args:
        process: The process where the inverse operations will be applied.
    """
    process.adj_begin()
    try:
        yield
    finally:
        process.adj_end()


def cat(*gates) -> Callable[[Any], Any]:
    """Concatenate gates.

    Create a new callable concatenating all the quantum gates.

    Example:

        .. code-block:: python

            from ket import *

            z_gate = cat(H, X, H)

            p = process()
            q = p.alloc()

            z_gate(q)

    Args:
        *gates: Quantum gates to concatenate.

    Returns:
        A new callable that concatenates the given quantum gates.
    """

    def inner(*args):
        for gate in gates:
            args = gate(*args)
            if not isinstance(args, tuple):
                args = (args,)

        if len(args) == 1:
            return args[0]
        return args

    return inner


def kron(*gates) -> Callable[[Any], Any]:
    """Gates tensor product.

    Create a new callable that with the tensor product of the given gates.

    Example:

        .. code-block:: python

            from ket import *

            p = process()
            a, b = p.alloc(2)

            HX = kron(H, X)

            HX(a, b) # Apply an Hadamard on qubit `a` and a Pauli X on qubit `b`

    Args:
        *gates: Quantum gates to tensor product.

    Returns:
        A new callable that represents the tensor product of the given quantum gates.
    """

    def inner(*args):
        return tuple(gate(arg) for gate, arg in zip(gates, args))

    return inner


@contextmanager
def around(
    gate: Callable[[Any], Any], *args, ket_process: Process | None = None, **kwargs
):
    r"""Applying and then reversing quantum gates.

    Apply the given quantum gate (:math:`U`) and then execute a code block (:math:`V`). After the
    code block is executed, the inverse of the gate (:math:`U^\dagger`) is applied, resulting in
    :math:`UVU^\dagger`.

    Example:

        .. code-block:: python

            from ket import *

            p = Process()
            a, b = p.alloc(2)

            bell = cat(kron(H, I), CNOT)

            with around(bell, a, b): # Apply bell(a, b)
                X(q)                 # Execute the code block
                                    # Apply adj(bell)(a, b)

    Args:
        gate: Quantum gate to apply.
        *arg: Qubits or classical parameters for the gate.
        ket_process: Explicitly specify the quantum process. If not provided, the process
                     is inferred from the qubits.
        **kwargs: Additional parameters for the quantum gate.

    """

    ket_process = _search_process(ket_process, args, kwargs)

    ket_process.ctrl_stack()
    gate(*args, **kwargs)
    ket_process.ctrl_unstack()
    try:
        yield
    finally:
        ket_process.ctrl_stack()
        adj(gate)(*args, ket_process=ket_process, **kwargs)
        ket_process.ctrl_unstack()


def measure(qubits: Quant) -> Measurement:
    """Measure the given qubits and return a measurement object.

    Args:
        qubits: Qubits to be measured.

    Returns:
        Object representing the measurement results.
    """
    if not isinstance(qubits, Quant):
        qubits = reduce(add, qubits)

    return Measurement(qubits)


def dump(qubits: Quant) -> QuantumState:
    """Obtain the quantum state snapshot.

    Args:
        qubits: Qubits to be observed.

    Returns:
        Object representing the quantum state.
    """
    if not isinstance(qubits, Quant):
        qubits = reduce(add, qubits)

    return QuantumState(qubits)


def sample(qubits: Quant, shots: int = 2048) -> Samples:
    """Get the quantum state measurement samples.

    Args:
        qubits: Qubits to be measured.
        shots: Number of measurement shots.

    Returns:
        Object representing the measurement samples.
    """
    if not isinstance(qubits, Quant):
        qubits = reduce(add, qubits)

    return Samples(qubits, int(shots))


def exp_value(hamiltonian: Hamiltonian | Pauli) -> ExpValue:
    """Calculate the expected value for a quantum state.

    Args:
        hamiltonian: Hamiltonian or Pauli operator for calculating the expected value.

    Returns:
        Object representing the expected value.
    """
    return ExpValue(hamiltonian)
