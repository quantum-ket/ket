"""Quantum integer data structure."""

# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0
from math import pi
from .base import Quant
from .gates import X, QFT, P
from .operations import ctrl, around, adj

__all__ = ["Qint"]


def _to_bin(num, n_bits):
    return f"{num & ((1 << n_bits) - 1):0{n_bits}b}"


def _set_int(qubits: Quant, state: int):
    state = _to_bin(state, len(qubits))
    for i, qubit in zip(state, qubits):
        if i == "1":
            X(qubit)


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


def _addi_qi(lhs: Quant, rhs: int | str, m=1, unsigned=False):
    n_lhs = len(lhs)
    if isinstance(rhs, int):
        if unsigned:
            n_rhs = rhs.bit_length()
            if n_rhs > n_lhs:
                raise RuntimeError(
                    "The number of bits in the right-hand side must be"
                    "less than or equal to the left-hand side."
                )

            rhs = f"{rhs:0{n_rhs}b}"
        else:
            n_rhs = n_lhs
            rhs = list(_to_bin(rhs, n_rhs))
    else:
        n_rhs = len(rhs)

    if n_lhs == 0:
        return

    offset = n_lhs - n_rhs + 1

    for j, q in enumerate(rhs):
        if q == "1":
            _rotation(j + offset, lhs[0], m)

    if n_rhs == n_lhs:
        _addi_qi(lhs[1:], rhs[1:], m)
    else:
        _addi_qi(lhs[1:], rhs, m)


def _addi(lhs, rhs, m: int = 1):
    with around(QFT, reversed(lhs.qubits), False):
        if isinstance(rhs, Qint):
            _addi_qq(
                lhs.qubits,
                rhs.qubits,
                m=m,
            )
        elif isinstance(rhs, int):
            _addi_qi(
                lhs.qubits,
                rhs,
                m=m,
                unsigned=lhs.unsigned,
            )
        else:
            raise TypeError("`other` must be a Qint or an integer.")


class Qint:  # pylint: disable=too-few-public-methods
    """A quantum integer data structure.

    This class represents an integer encoded within a quantum register.
    It provides a high-level interface for quantum arithmetic, supporting
    in-place addition and subtraction operations.

    Args:
        qubits: The quantum register to allocate for this integer.
        number: The initial classical integer value to set
            the quantum register to. Defaults to 0.
        unsigned: Flag indicating whether the integer
            should be treated as unsigned. Defaults to False.
    """

    def __init__(self, qubits: Quant, number: int = 0, unsigned: bool = False):

        self.qubits = qubits
        self.unsigned = unsigned
        self.process = qubits.process
        _set_int(qubits, number)

    def _get_ket_process(self):
        return self.process

    def __iadd__(self, other):
        _addi(self, other)
        return self

    __isub__ = adj(__iadd__)

    def mul(self: Quant, other, result):
        """Performs quantum multiplication.

        Multiplies the Qint ``self`` (multiplier) by ``other``
        (multiplicand) and accumulates the product into the ``result`` Qint.

        Args:
            self: The quantum register acting as the multiplier.
            other: The multiplicand (can be a classical integer
                or another Qint).
            result: The target Qint where the product
                is stored (accumulated in-place).
        """
        if isinstance(result, Quant):
            result = Qint(result)

        for i, q in enumerate(reversed(self.qubits)):
            ctrl(q, _addi)(result, other, m=2**i)
