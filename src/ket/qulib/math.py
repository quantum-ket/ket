"""Quantum arithmetic operations for quantum states."""

# SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

from math import pi
from ..base import Quant
from ..gates import QFT, P, X
from ..operations import ctrl, around

__all__ = [
    "addi",
    "mul",
    "set_int",
]


def _rotation(l, q, m=1):
    P(2 * pi / (2**l / m), q)


def _addi_qq(lhs: Quant, rhs: Quant, m=1):
    n_lhs = len(lhs)
    n_rhs = len(rhs)
    if n_rhs > n_lhs:
        raise RuntimeError(
            "The number of qubits in the right-hand side must be"
            "less than or equal to the left-hand side."
        )
    if n_lhs == 0:
        return

    offset = n_lhs - n_rhs + 1

    for j, q in enumerate(rhs):
        ctrl(q, _rotation)(j + offset, lhs[0], m)

    if n_rhs == n_lhs:
        _addi_qq(lhs[1:], rhs[1:], m)
    else:
        _addi_qq(lhs[1:], rhs, m)


def _addi_qi(lhs: Quant, rhs: int, m=1):
    n_lhs = len(lhs)
    n_rhs = rhs.bit_length()
    if n_rhs > n_lhs:
        raise RuntimeError(
            "The number of bits in the right-hand side must be"
            "less than or equal to the left-hand side."
        )
    if n_lhs == 0:
        return

    offset = n_lhs - n_rhs + 1

    for j, q in enumerate(f"{rhs:0{n_rhs}b}"):
        if q == "1":
            _rotation(j + offset, lhs[0], m)

    if n_rhs == n_lhs:
        _addi_qi(lhs[1:], rhs & ((1 << n_rhs - 1) - 1), m)
    else:
        _addi_qi(lhs[1:], rhs, m)


def addi(lhs: Quant, rhs: Quant | int, multiplier: int = 1):
    """Add two quantum states or a quantum state and an integer.

    The result is stored in the left-hand side quantum state.

    The multiplier is used to scale the right-hand side.

    Args:
        lhs: Left-hand side quantum state.
        rhs: Right-hand side quantum state or integer to add.
        multiplier: Multiplier for the addition (default is 1).
    """

    with around(QFT, reversed(lhs), False):
        if isinstance(rhs, Quant):
            _addi_qq(lhs, rhs, multiplier)
        elif isinstance(rhs, int):
            _addi_qi(lhs, rhs, multiplier)
        else:
            raise TypeError("Right-hand side must be a Quant or an integer.")


def mul(lhs: Quant, rhs: Quant | int, result: Quant):
    """Multiply two quantum states.

    The result is stored in the result quantum state.

    Args:
        lhs: Left-hand side quantum state.
        rhs: Right-hand side quantum state or integer to add.
        result: Quantum state to store the result.
    """

    for i, q in enumerate(reversed(lhs)):
        ctrl(q, addi)(result, rhs, 2**i)


def set_int(qubits: Quant, state: int):
    r"""Set the state of a quantum register.

    If the qubits are in the state :math:`\ket{0\dots0}`, this function will
    set them to the state :math:`\ket{\texttt{state}}`.

    Args:
        state: Integer representing the state to set.
        qubits: Quantum register to modify.
    """

    for i, qubit in zip(f"{state:0{len(qubits)}b}", qubits):
        if i == "1":
            X(qubit)
