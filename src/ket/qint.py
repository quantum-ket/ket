"""Quantum integer data structure."""

# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=duplicate-code,protected-access

from functools import partial
from math import pi
from .base import Quant
from .gates import X, QFT, P, is_permutation
from .operations import C, control, ctrl, around, adj, undo

__all__ = ["Qreal", "Qint"]


def _to_bin(num, n_bits):
    """Converts an integer to its binary string representation with a fixed number of bits."""
    return f"{num & ((1 << n_bits) - 1):0{n_bits}b}"


def _to_signed(value, n_bits):
    value = value & ((1 << n_bits) - 1)
    sing_bit = 1 << (n_bits - 1)
    if value & sing_bit:
        return value - (1 << n_bits)
    return value


def _set_int(qubits: Quant, state: int):
    """Initializes a quantum register to a specific classical integer state using X gates."""
    state = _to_bin(state, len(qubits))
    for i, qubit in zip(state, qubits):
        if i == "1":
            X(qubit)


def _rotation(l, q, m=1):
    """Applies a phase rotation for the Quantum Fourier Transform addition."""
    P(2 * pi / (2**l / m), q)


def _addi_qq(lhs: Quant, rhs: Quant, m=1):
    """In-place quantum-quantum addition in the Fourier basis."""
    n_lhs = len(lhs)
    n_rhs = len(rhs)
    if n_rhs > n_lhs:
        raise RuntimeError(
            "The number of qubits in the right-hand side must be "
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


def _addi_qi(lhs: Quant, rhs: int | str, m=1):
    """In-place quantum-classical addition in the Fourier basis."""
    n_lhs = len(lhs)
    if isinstance(rhs, int):
        n_rhs = rhs.bit_length()

        if n_rhs > n_lhs:
            raise RuntimeError(
                "The number of bits in the right-hand side must be "
                "less than or equal to the left-hand side."
            )
        if rhs >= 0:
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


@is_permutation
def _addi(lhs, rhs, m: int = 1):
    """General in-place addition using the QFT."""
    with around(QFT, reversed(lhs), False):
        if isinstance(rhs, Qreal):
            _addi_qq(lhs, rhs, m=m)
        elif isinstance(rhs, int):
            _addi_qi(lhs, rhs, m=m)
        else:
            raise TypeError("`other` must be a Qint, Qreal, or an integer.")


class Qreal(Quant):  # pylint: disable=too-few-public-methods
    """A quantum real data structure using fixed-point representation.

    This class represents a real number encoded within a quantum register.
    It provides a high-level interface for quantum arithmetic, supporting
    in-place addition, subtraction, multiplication, and division.

    Args:
        qubits: The quantum register to allocate for this number.
        exp: The exponent defining the fixed-point scale (value x 2**exp).
        number: The initial classical float value to set the quantum register to.
                Defaults to 0.0.
    """

    def __init__(self, qubits: Quant, exp, number: float = 0.0):
        super().__init__(
            qubits=qubits.qubits, process=qubits.ket_process, source=qubits
        )
        self.exp = exp
        number = round(number * 2**exp)
        _set_int(qubits, number)

    def __iadd__(self, other):
        if isinstance(other, Qreal):
            m = self.exp - other.exp
        else:
            other = round(other * 2**self.exp)
            m = 0

        _addi(self, other, m=2**m)
        return self

    __isub__ = adj(__iadd__)

    def copy(self):
        """Copies the quantum state into a new Qreal register."""
        other = self.ket_process.alloc_aux(len(self))

        def inner_copy(other):
            for s, o in zip(self, other):
                C(X)(s, o)

        return Qreal(undo(inner_copy, other), exp=self.exp)

    def __add__(self, other):
        result = self.copy()

        def inner_add(result):
            result += other

        return undo(inner_add, result)

    def __sub__(self, other):
        result = self.copy()

        def inner_sub(result):
            result -= other

        return undo(inner_sub, result)

    def mul(self, other, result):
        """Performs quantum multiplication.

        Multiplies the Qreal ``self`` (multiplier) by ``other``
        (multiplicand) and accumulates the product into the ``result`` Qreal.

        Args:
            other: The multiplicand (can be a classical scalar or another Qreal).
            result: The target quantum register where the product is accumulated.
        """
        if isinstance(other, Qreal):
            m = self.exp - other.exp
        else:
            other = round(other)
            m = 0

        for i, q in enumerate(reversed(self)):
            ctrl(q, _addi)(result, other, m=2 ** (i + m))

    def __mul__(self, other):
        result_quant = self.ket_process.alloc_aux(len(self))
        new_exp = self.exp + other.exp if isinstance(other, Qreal) else self.exp

        raw_result = undo(partial(self.mul, other), result_quant)
        return Qreal(raw_result, exp=new_exp)

    def __truediv__(self, other):
        result_quant = self.ket_process.alloc_aux(len(self))
        new_exp = self.exp - other.exp if isinstance(other, Qreal) else self.exp

        raw_result = undo(partial(adj(self.mul), other), result_quant)
        return Qreal(raw_result, exp=new_exp)

    def __neg__(self):
        def neg(qint):
            X(qint)
            qint += 1

        result = self.copy()
        return undo(neg, result)

    def __lt__(self, other):
        result = self.copy()

        def lt(result):
            result -= other

        return undo(lt, result)[0]

    def __gt__(self, other):
        result = -self + other
        return result[0]

    def __ge__(self, other):
        return undo(X, (self < other))

    def __le__(self, other):
        return undo(X, (self > other))

    def __eq__(self, other):
        result = self.ket_process.alloc_aux()

        def eq(result):
            with control(Quant.__eq__(self - other, 0)):
                X(result)

        return undo(eq, result)

    def __ne__(self, value):
        return undo(X, self == value)

    def postprocessing(self):
        """Returns a function to convert the internal integer state back to a float."""

        def postprocessing(exp, size, value):
            value = _to_signed(value, size)

            return value / (2**exp)

        return partial(postprocessing, int(self.exp), len(self))

    def dump_format(self):
        def dump_format(pp, state):
            return str(pp(state))

        return partial(dump_format, self.postprocessing())


class Qint(Qreal):
    """A quantum integer data structure.

    A specialized case of Qreal where the exponent is strictly 0.

    Args:
        qubits: The quantum register to allocate for this number.
        number: The initial classical integer value. Defaults to 0.
    """

    def __init__(self, qubits, number: int = 0):
        super().__init__(qubits, exp=0, number=number)

    def postprocessing(self):
        """Returns a function to convert the internal integer state back to a int."""

        def postprocessing(size, value):
            return _to_signed(value, size)

        return partial(postprocessing, len(self))
