from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from fractions import Fraction
from functools import partial
from math import pi
from typing import Callable, Literal
from ctypes import c_size_t

from .clib.libket import (
    HADAMARD,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    ROTATION_X,
    ROTATION_Y,
    ROTATION_Z,
)

from .base import quant

__all__ = [
    "control",
    "ctrl",
    "inverse",
    "adj",
    "X",
    "Y",
    "Z",
    "H",
    "RX",
    "RY",
    "RZ",
]


@contextmanager
def control(qubits: quant):
    process = qubits.process
    process.ctrl_push(
        (c_size_t * len(qubits.qubits))(*qubits.qubits),
        len(qubits.qubits),
    )
    try:
        yield
    finally:
        process.ctrl_pop()


def ctrl(ctrl_qubits: quant, gate, *args, **kwargs):
    with control(ctrl_qubits):
        gate(*args, **kwargs)


@contextmanager
def inverse(process):
    process.adj_begin()
    try:
        yield
    finally:
        process.adj_end()


def adj(gate):
    def inner(*args, **kwargs):
        process = None
        for arg in args:
            if hasattr(arg, "_get_ket_process"):
                arg_process = arg._get_ket_process()
                if process is not None and process is not arg_process:
                    raise ValueError("parameter with different Ket processes")
                process = arg_process
        for arg in kwargs.values():
            if hasattr(arg, "_get_ket_process"):
                arg_process = arg._get_ket_process()
                if process is not None and process is not arg_process:
                    raise ValueError("parameter with different Ket processes")
                process = arg_process

        if process is None:
            raise ValueError("no Ket process found in parameters")

        with inverse(process):
            gate(*args, **kwargs)

    return inner


def X(qubits: quant) -> quant:
    for qubit in qubits.qubits:
        qubits.process.apply_gate(PAULI_X, 1, 1, 0.0, qubit)
    return qubits


def Y(qubits: quant) -> quant:
    for qubit in qubits.qubits:
        qubits.process.apply_gate(PAULI_Y, 1, 1, 0.0, qubit)
    return qubits


def Z(qubits: quant) -> quant:
    for qubit in qubits.qubits:
        qubits.process.apply_gate(PAULI_Z, 1, 1, 0.0, qubit)
    return qubits


def H(qubits: quant) -> quant:
    for qubit in qubits.qubits:
        qubits.process.apply_gate(HADAMARD, 1, 1, 0.0, qubit)
    return qubits


def RX(theta: float, qubits: quant | None = None) -> quant | Callable[[quant], quant]:
    top, bottom = Fraction(theta / pi).limit_denominator().as_integer_ratio()
    use_fraction = abs(pi * top / bottom - theta) < 1e-14
    params = (top, bottom, 0.0) if use_fraction else (0, 0, theta)

    def inner(qubits: quant) -> quant:
        for qubit in qubits.qubits:
            qubits.process.apply_gate(ROTATION_X, *params, qubit)
        return qubits

    if qubits is None:
        return partial(inner, theta)
    return inner(qubits)


def RY(theta: float, qubits: quant | None = None) -> quant | Callable[[quant], quant]:
    top, bottom = Fraction(theta / pi).limit_denominator().as_integer_ratio()
    use_fraction = abs(pi * top / bottom - theta) < 1e-14
    params = (top, bottom, 0.0) if use_fraction else (0, 0, theta)

    def inner(qubits: quant) -> quant:
        for qubit in qubits.qubits:
            qubits.process.apply_gate(ROTATION_Y, *params, qubit)
        return qubits

    if qubits is None:
        return partial(inner, theta)
    return inner(qubits)


def RZ(theta: float, qubits: quant | None = None) -> quant | Callable[[quant], quant]:
    top, bottom = Fraction(theta / pi).limit_denominator().as_integer_ratio()
    use_fraction = abs(pi * top / bottom - theta) < 1e-14
    params = (top, bottom, 0.0) if use_fraction else (0, 0, theta)

    def inner(qubits: quant) -> quant:
        for qubit in qubits.qubits:
            qubits.process.apply_gate(ROTATION_Z, *params, qubit)
        return qubits

    if qubits is None:
        return partial(inner, theta)
    return inner(qubits)
