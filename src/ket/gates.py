"""Quantum gate definitions"""
from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from fractions import Fraction
from functools import reduce
from math import pi
from operator import add
from typing import Callable

from .clib.libket import (
    HADAMARD,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    ROTATION_X,
    ROTATION_Y,
    ROTATION_Z,
    PHASE_SHIFT,
)

from .base import Quant
from .operations import ctrl, adj, cat, kron, around

__all__ = [
    "I",
    "X",
    "Y",
    "Z",
    "H",
    "RX",
    "RY",
    "RZ",
    "PHASE",
    "S",
    "T",
    "SD",
    "TD",
    "CNOT",
    "SWAP",
    "RXX",
    "RZZ",
    "RYY",
    "flip_to_control",
    "phase_oracle",
]


def I(qubits: Quant) -> Quant:  # pylint: disable=invalid-name
    """Apply the Identity gate."""
    qubits = reduce(add, qubits)

    return qubits


def X(qubits: Quant) -> Quant:  # pylint: disable=invalid-name
    """Apply the Pauli X gate."""
    qubits = reduce(add, qubits)
    for qubit in qubits.qubits:
        qubits.process.apply_gate(PAULI_X, 1, 1, 0.0, qubit)
    return qubits


def Y(qubits: Quant) -> Quant:  # pylint: disable=invalid-name
    """Apply the Pauli Y gate."""

    qubits = reduce(add, qubits)
    for qubit in qubits.qubits:
        qubits.process.apply_gate(PAULI_Y, 1, 1, 0.0, qubit)
    return qubits


def Z(qubits: Quant) -> Quant:  # pylint: disable=invalid-name
    """Apply the Pauli Z gate."""

    qubits = reduce(add, qubits)
    for qubit in qubits.qubits:
        qubits.process.apply_gate(PAULI_Z, 1, 1, 0.0, qubit)
    return qubits


def H(qubits: Quant) -> Quant:  # pylint: disable=invalid-name
    """Apply the Hadamard gate."""

    qubits = reduce(add, qubits)
    for qubit in qubits.qubits:
        qubits.process.apply_gate(HADAMARD, 1, 1, 0.0, qubit)
    return qubits


def RX(  # pylint: disable=invalid-name
    theta: float, qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:
    """Apply a x-axes rotation gate."""
    top, bottom = Fraction(theta / pi).limit_denominator().as_integer_ratio()
    use_fraction = abs(pi * top / bottom - theta) < 1e-14
    params = (top, bottom, 0.0) if use_fraction else (0, 0, theta)

    def inner(qubits: Quant) -> Quant:
        qubits = reduce(add, qubits)
        for qubit in qubits.qubits:
            qubits.process.apply_gate(ROTATION_X, *params, qubit)
        return qubits

    if qubits is None:
        return inner
    return inner(qubits)


def RY(  # pylint: disable=invalid-name
    theta: float, qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:
    """Apply a y-axes rotation gate."""

    top, bottom = Fraction(theta / pi).limit_denominator().as_integer_ratio()
    use_fraction = abs(pi * top / bottom - theta) < 1e-14
    params = (top, bottom, 0.0) if use_fraction else (0, 0, theta)

    def inner(qubits: Quant) -> Quant:
        qubits = reduce(add, qubits)
        for qubit in qubits.qubits:
            qubits.process.apply_gate(ROTATION_Y, *params, qubit)
        return qubits

    if qubits is None:
        return inner
    return inner(qubits)


def RZ(  # pylint: disable=invalid-name
    theta: float, qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:
    """Apply a y-axes rotation gate."""

    top, bottom = Fraction(theta / pi).limit_denominator().as_integer_ratio()
    use_fraction = abs(pi * top / bottom - theta) < 1e-14
    params = (top, bottom, 0.0) if use_fraction else (0, 0, theta)

    def inner(qubits: Quant) -> Quant:
        qubits = reduce(add, qubits)
        for qubit in qubits.qubits:
            qubits.process.apply_gate(ROTATION_Z, *params, qubit)
        return qubits

    if qubits is None:
        return inner
    return inner(qubits)


def PHASE(  # pylint: disable=invalid-name
    theta: float, qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:
    """Apply a phase gate."""
    top, bottom = Fraction(theta / pi).limit_denominator().as_integer_ratio()
    use_fraction = abs(pi * top / bottom - theta) < 1e-14
    params = (top, bottom, 0.0) if use_fraction else (0, 0, theta)

    def inner(qubits: Quant) -> Quant:
        qubits = reduce(add, qubits)
        for qubit in qubits.qubits:
            qubits.process.apply_gate(PHASE_SHIFT, *params, qubit)
        return qubits

    if qubits is None:
        return inner
    return inner(qubits)


S = PHASE(pi / 2)
SD = adj(S)
T = PHASE(pi / 4)
TD = adj(T)


def CNOT(  # pylint: disable=invalid-name
    control_qubit: Quant, target_qubit: Quant
) -> tuple[Quant, Quant]:
    """Apply a controlled not gate."""
    control_qubit = reduce(add, control_qubit)
    target_qubit = reduce(add, target_qubit)
    return ctrl(control_qubit, X)(target_qubit)


def SWAP(  # pylint: disable=invalid-name
    qubit_a: Quant, qubit_b: Quant
) -> tuple[Quant, Quant]:
    """Apply a SWAP gate."""
    return cat(CNOT, lambda a, b: (b, a), CNOT, lambda a, b: (b, a), CNOT)(
        qubit_a, qubit_b
    )


def RXX(  # pylint: disable=invalid-name
    theta: float, qubits_a: Quant | None, qubits_b: Quant | None
) -> tuple[Quant, Quant] | Callable[[Quant, Quant], tuple[Quant, Quant]]:
    """Apply a XX rotation gate."""

    def inner(qubits_a: Quant, qubits_b: Quant) -> tuple[Quant, Quant]:
        for qubit_a, qubit_b in zip(qubits_a, qubits_b):
            with around(cat(kron(H, H), CNOT), qubit_a, qubit_b):
                RZ(theta, qubit_b)

        return qubits_a, qubits_b

    if qubits_a is None and qubits_b is None:
        return inner
    return inner(qubits_a, qubits_b)


def RZZ(  # pylint: disable=invalid-name
    theta: float, qubits_a: Quant | None, qubits_b: Quant | None
) -> tuple[Quant, Quant] | Callable[[Quant, Quant], tuple[Quant, Quant]]:
    """Apply a ZZ rotation gate."""

    def inner(qubits_a: Quant, qubits_b: Quant) -> tuple[Quant, Quant]:
        for qubit_a, qubit_b in zip(qubits_a, qubits_b):
            with around(CNOT, qubit_a, qubit_b):
                RZ(theta, qubit_b)

        return qubits_a, qubits_b

    if qubits_a is None and qubits_b is None:
        return inner
    return inner(qubits_a, qubits_b)


def RYY(  # pylint: disable=invalid-name
    theta: float, qubits_a: Quant | None, qubits_b: Quant | None
) -> tuple[Quant, Quant] | Callable[[Quant, Quant], tuple[Quant, Quant]]:
    """Apply a YY rotation gate."""

    def inner(qubits_a: Quant, qubits_b: Quant) -> tuple[Quant, Quant]:
        for qubit_a, qubit_b in zip(qubits_a, qubits_b):
            with around(cat(kron(RX(pi / 2), RX(pi / 2)), CNOT), qubit_a, qubit_b):
                RZ(theta, qubit_b)
        return qubits_a, qubits_b

    if qubits_a is None and qubits_b is None:
        return inner
    return inner(qubits_a, qubits_b)


def flip_to_control(
    control_state: int | list[int], qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:
    """Flip qubits |control_state> -> |1...1>."""

    def inner(qubits: Quant) -> Quant:
        qubits = reduce(add, qubits)

        length = len(qubits)
        if hasattr(control_state, "__iter__"):
            if len(control_state) != length:
                raise ValueError(
                    f"'to' received a list of length {len(control_state)} to use on {length} qubits"
                )
        else:
            if length < control_state.bit_length():
                raise ValueError(
                    f"To flip with control_state={control_state} "
                    f"you need at least {control_state.bit_length()} qubits"
                )

            state = [int(i) for i in f"{{:0{length}b}}".format(control_state)]

        for i, qubit in zip(state, qubits):
            if i == 0:
                X(qubit)
        return qubits

    if qubits is None:
        return inner
    return inner(qubits)


def phase_oracle(
    state: int, qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:
    """Apply a -1 phase on the state."""

    def inner(qubits: Quant) -> Quant:
        init, last = qubits[:-1], qubits[-1]
        with around(flip_to_control(state >> 1), init):
            with around(lambda q: X(q) if state & 1 == 0 else None, last):
                ctrl(init, Z)(last)

    if qubits is None:
        return inner
    return inner(qubits)
