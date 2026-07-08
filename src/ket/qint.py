"""Quantum integer and fixed-point real number data structures.

This module provides :class:`~ket.qint.Qreal` and :class:`~ket.qint.Qint`,
two high-level quantum data structures that wrap a :class:`~ket.base.Quant`
register and expose Python-style arithmetic operators as quantum operations.

Both types use a **QFT-based arithmetic** approach:

- In-place addition/subtraction are implemented via phase rotations in the
  Quantum Fourier Transform basis.
- Quantum comparisons (:meth:`~ket.qint.Qreal.__lt__`, etc.) produce
  ancilla qubit results that are cleaned up automatically via
  :func:`~ket.operations.undo`.
- Multiplication and division are built from repeated addition in the Fourier
  basis.

All operations produce **uncomputed** intermediate registers where possible,
keeping the circuit depth manageable.

Typical Usage
-------------

.. code-block:: python

    from ket import Process, measure

    p = Process()
    q = p.alloc(8)

    # Integer arithmetic
    qi = q.as_int(7)    # initialize register to |7⟩
    qi += 3             # quantum addition: |7⟩ → |10⟩
    print(measure(qi).value)   # 10
    # 10

.. code-block:: python

    from ket import Process, measure

    p = Process()
    q = p.alloc(8)

    # Fixed-point real arithmetic with 4 bits of fractional precision
    qr = q.as_real(4, 1.5)   # step size 2**-4 = 0.0625
    qr += 0.25
    print(measure(qr).value)  # 1.75
    # 1.75
"""

# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=duplicate-code,protected-access

from functools import partial
from math import pi
from typing import Any
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
        if isinstance(rhs, Quant):
            _addi_qq(lhs, rhs, m=m)
        elif isinstance(rhs, int):
            _addi_qi(lhs, rhs, m=m)
        else:
            raise TypeError("`rhs` must be a Qint, Qreal, or an integer.")


class Qreal(Quant):  # pylint: disable=too-few-public-methods
    """A quantum register interpreted as a fixed-point signed real number.

    Extends :class:`~ket.base.Quant` to support in-place arithmetic
    (:math:`+=`, :math:`-=`), non-in-place arithmetic (:math:`+`, :math:`-`,
    :math:`\\times`, :math:`/`), and comparisons (:math:`<`, :math:`>`,
    :math:`\\leq`, :math:`\\geq`, :math:`=`, :math:`\\neq`) as quantum
    operations.

    The register stores a **two's-complement signed integer** :math:`n` that
    represents the real value

    .. math::

        x = \\frac{n}{2^{\\texttt{exp}}}

    - A positive ``exp`` gives finer fractional resolution
      (:math:`2^{-\\texttt{exp}}` per step).
    - A negative ``exp`` gives coarser resolution but a larger range.

    Example:

        .. code-block:: python

            from ket import Process, measure

            p = Process()
            q = p.alloc(8)            # 8-qubit register

            # Fixed-point with 4-bit fractional precision (step = 1/16)
            qr = q.as_real(4, 1.5)    # stored integer = round(1.5 * 16) = 24
            qr += 0.5                 # 24 + 8 = 32 → x = 32/16 = 2.0
            print(measure(qr).value)  # 2.0
            # 2.0

    Args:
        qubits: The quantum register to wrap.
        exp: Fixed-point exponent. The register stores ``round(number * 2**exp)``.
        number: Initial real value to encode. Defaults to ``0.0``.
    """

    def __init__(self, qubits: Quant, exp, number: float = 0.0):
        super().__init__(
            qubits=qubits.qubits, process=qubits.ket_process, source=qubits
        )
        self.exp = exp
        number = round(number * 2**exp)
        _set_int(qubits, number)

    def __iadd__(self, other: Any):
        if not isinstance(other, (int, float, Quant)):
            return NotImplemented

        if isinstance(other, Qreal):
            m = self.exp - other.exp
        else:
            other = round(other * 2**self.exp)
            m = 0

        _addi(self, other, m=2**m)
        return self

    __isub__ = adj(__iadd__)

    def copy(self):
        """Create a copy of this register in a fresh auxiliary register.

        Allocates a new auxiliary qubit register of the same size and uses
        CNOT gates to copy the state qubit-by-qubit. The copy is wrapped
        in :func:`~ket.operations.undo` so that the auxiliary register is
        automatically uncomputed when the returned object goes out of scope.

        Returns:
            A new :class:`~ket.qint.Qreal` wrapping
            an auxiliary register that holds a copy of this register's state.
        """
        other = self.ket_process.alloc_aux(len(self))

        def inner_copy(other):
            for s, o in zip(self, other):
                C(X)(s, o)

        return Qreal(undo(inner_copy, other), exp=self.exp)

    def __add__(self, other: Any):
        if not isinstance(other, (int, float, Quant)):
            return NotImplemented

        result = self.copy()

        def inner_add(result):
            result += other

        return Qreal(undo(inner_add, result), exp=self.exp)

    def __sub__(self, other: Any):
        if not isinstance(other, (int, float, Quant)):
            return NotImplemented

        result = self.copy()

        def inner_sub(result):
            result -= other

        return Qreal(undo(inner_sub, result), exp=self.exp)

    def mul(self, other, result):
        """Quantum multiplication: accumulate ``self * other`` into ``result``.

        Performs an in-place multiply-accumulate: each bit of ``self``
        (the multiplier) controls an addition of the appropriately shifted
        ``other`` (the multiplicand) into ``result``.

        This method is used internally by :meth:`~ket.qint.Qreal.__mul__` and
        :meth:`~ket.qint.Qreal.__truediv__`.

        Args:
            other: The multiplicand.
                If a :class:`~ket.qint.Qreal`, the fixed-point exponents are
                reconciled automatically.
            result: The target register where the
                product is accumulated. Must be initialized to 0.
        """
        if isinstance(other, Qreal):
            m = self.exp - other.exp
        else:
            other = round(other)
            m = 0

        for i, q in enumerate(reversed(self)):
            ctrl(q, _addi)(result, other, m=2 ** (i + m))

    def __mul__(self, other: Any):
        if not isinstance(other, (int, float, Quant)):
            return NotImplemented

        result_quant = self.ket_process.alloc_aux(len(self))
        new_exp = self.exp + other.exp if isinstance(other, Qreal) else self.exp

        raw_result = undo(partial(self.mul, other), result_quant)
        return Qreal(raw_result, exp=new_exp)

    def __truediv__(self, other: Any):
        if not isinstance(other, (int, float, Quant)):
            return NotImplemented

        result_quant = self.ket_process.alloc_aux(len(self))
        new_exp = self.exp - other.exp if isinstance(other, Qreal) else self.exp

        raw_result = undo(partial(adj(self.mul), other), result_quant)
        return Qreal(raw_result, exp=new_exp)

    def __neg__(self):
        def neg(qint):
            X(qint)
            qint += 1

        result = self.copy()
        return Qreal(undo(neg, result), exp=self.exp)

    def __lt__(self, other: Any):
        if not isinstance(other, (int, float, Quant)):
            return NotImplemented

        result = self.copy()

        def lt(result):
            result -= other

        return undo(lt, result)[0]

    def __gt__(self, other: Any):
        if not isinstance(other, (int, float, Quant)):
            return NotImplemented

        result = -self + other
        return result[0]

    def __ge__(self, other: Any):
        if not isinstance(other, (int, float, Quant)):
            return NotImplemented

        return undo(X, (self < other))

    def __le__(self, other: Any):
        if not isinstance(other, (int, float, Quant)):
            return NotImplemented

        return undo(X, (self > other))

    def __eq__(self, other: Any):
        if not isinstance(other, (int, float, Quant)):
            return NotImplemented

        cond = Quant.__eq__(self - other, 0)
        result = self.ket_process.alloc_aux()

        def eq(result):
            with control(cond):
                X(result)

        return undo(eq, result)

    def __ne__(self, other: Any):
        if not isinstance(other, (int, float, Quant)):
            return NotImplemented

        return undo(X, self == other)

    def postprocessing(self):
        """Return a function that converts the raw qubit measurement to a float.

        Used internally by :func:`~ket.operations.measure` and
        :func:`~ket.operations.dump` to convert the measured integer
        back to its fixed-point float representation.

        Returns:
            A function that accepts the raw unsigned
            integer measurement result and returns the corresponding float
            value (with sign extension for two's-complement).
        """

        def postprocessing(exp, size, value):
            value = _to_signed(value, size)

            return value / (2**exp)

        return partial(postprocessing, int(self.exp), len(self))

    def dump_format(self):
        def dump_format(pp, state):
            return str(pp(state))

        return partial(dump_format, self.postprocessing())


class Qint(Qreal):
    """A quantum register interpreted as a signed integer.

    A specialization of :class:`~ket.qint.Qreal` with ``exp=0``, meaning the
    stored integer is the value directly (no fractional scaling). Uses
    two's-complement representation for signed arithmetic.

    Example:

        .. code-block:: python

            from ket import Process, measure

            p = Process()
            q = p.alloc(4)

            qi = q.as_int(-3)   # two's-complement encoding
            qi += 5
            print(measure(qi).value)   # 2
            # 2

    Args:
        qubits: The quantum register to wrap.
        number: Initial integer value to encode. Defaults to ``0``.
    """

    def __init__(self, qubits, number: int = 0):
        super().__init__(qubits, exp=0, number=number)

    def postprocessing(self):
        """Return a function that converts a raw measurement to a signed integer.

        Returns:
            A function that accepts the unsigned integer
            measurement result and returns the corresponding signed (two's
            complement) integer value.
        """

        def postprocessing(size, value):
            return _to_signed(value, size)

        return partial(postprocessing, len(self))
