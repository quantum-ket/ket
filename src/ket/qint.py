"""Quantum integer and fixed-point real number data structures.

This module provides :class:`~ket.qint.Qreal` and :class:`~ket.qint.Qint`,
two high-level quantum data structures that wrap a :class:`~ket.base.Quant`
register and expose Python-style arithmetic operators as quantum operations.

Both types use a **QFT-based arithmetic** approach:

- In-place addition/subtraction are implemented via phase rotations in the
  Quantum Fourier Transform basis.
- Quantum comparisons (``<``, ``>``, ``==``, and so on) produce
  auxiliary qubit results that are cleaned up automatically via
  :func:`~ket.operations.undo`.
- Multiplication is built from repeated addition in the Fourier basis, and
  division from repeated subtraction (restoring long division).

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
from .operations import control, ctrl, around, adj, undo

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


@is_permutation
def _addi_signed(lhs: Quant, rhs: Quant | int, m: int = 1):
    """In-place addition of a signed (two's-complement) right-hand side.

    :func:`_addi` fills the missing high bits of a shorter ``rhs`` register
    with 0, which is only right when it is not negative. A negative ``rhs``
    of :math:`k` qubits holds :math:`r - 2^k`, where :math:`r` is its
    unsigned value, so this subtracts :math:`2^k` when its sign qubit is 1.
    """
    _addi(lhs, rhs, m=m)
    if isinstance(rhs, Quant) and len(rhs) < len(lhs):
        with control(rhs[0]):
            adj(_addi)(lhs, 2 ** len(rhs), m=m)


def _negate(qubits: Quant):
    """Two's-complement negation in place."""
    X(qubits)
    _addi(qubits, 1)


def _divide(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    workspace: Quant,
    dividend: Quant,
    divisor: Quant | int,
    shift: int,
    divisor_negative: bool = False,
):
    """Signed division, truncated toward zero, by restoring long division.

    Computes ``trunc(dividend * 2**shift / divisor)`` into the first
    ``len(dividend) + max(shift, 0)`` qubits of ``workspace``, which must have
    :func:`_division_size` qubits, all in state 0. The dividend and a quantum
    divisor are signed (two's complement) and are not changed. An ``int``
    divisor is the magnitude of a classical divisor, whose sign is
    ``divisor_negative``.
    """
    n = len(dividend)
    num_q = n + max(shift, 0)  # quotient bits
    quantum = isinstance(divisor, Quant)
    num_d = len(divisor) + max(-shift, 0) if quantum else divisor.bit_length()

    quotient = workspace[:num_q]
    remainder = workspace[num_q : 2 * num_q + num_d]

    # remainder = |dividend| * 2**shift
    _copy_abs(dividend, remainder[num_d : num_d + n])

    if quantum:
        # magnitude = |divisor| * 2**(-shift)
        magnitude = workspace[2 * num_q + num_d :]
        _copy_abs(divisor, magnitude[: len(divisor)])
    else:
        magnitude = divisor

    # One quotient bit per step, most significant first: subtract the divisor
    # from a window of the remainder, and add it back if the result is
    # negative. The quotient bit is 1 when the subtraction is kept.
    # (The window starts below 2 * divisor, so the difference fits in it and
    # its first qubit is the sign.)
    for j in range(num_q):
        window = remainder[j : j + num_d + 1]
        bit = quotient[j]
        adj(_addi)(window, magnitude)
        with control(window[0]):  # bit = 1 if negative
            X(bit)
        with control(bit):
            _addi(window, magnitude)
        X(bit)

    # The quotient is negative when exactly one operand is.
    with control(dividend[0]):
        _negate(quotient)
    if quantum:
        with control(divisor[0]):
            _negate(quotient)
    elif divisor_negative:
        _negate(quotient)


def _copy_abs(source: Quant, target: Quant):
    """Copy the absolute value of a two's-complement register to ``target``.

    ``target`` has ``len(source)`` qubits in state 0. The result is unsigned,
    so the most negative value, ``-2**(n-1)``, is copied correctly.
    """
    for src, dst in zip(source, target):
        with control(src):
            X(dst)
    with control(source[0]):
        _negate(target)


def _division_size(n: int, divisor: Quant | int, shift: int) -> int:
    """Number of workspace qubits for :func:`_divide`."""
    num_q = n + max(shift, 0)
    if isinstance(divisor, Quant):
        num_d = len(divisor) + max(-shift, 0)
        return 2 * num_q + 2 * num_d
    return 2 * num_q + divisor.bit_length()


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

    def copy(self, depends_on=None):
        return Qreal(super().copy(depends_on), exp=self.exp)

    def __iadd__(self, other: Any):
        """In-place addition: ``self += other``.

        ``other`` can be a number or another :class:`Qint`/:class:`Qreal` with
        at most as many qubits. The result wraps around on overflow (two's
        complement). ``self -= other`` is the inverse operation.
        """
        if not isinstance(other, (int, float, Quant)):
            return NotImplemented

        if isinstance(other, Qreal):
            m = self.exp - other.exp
        else:
            other = round(other * 2**self.exp)
            m = 0

        _addi_signed(self, other, m=2**m)
        return self

    __isub__ = adj(__iadd__)

    def __add__(self, other: Any):
        """Addition: ``self + other``, into a new auxiliary register.

        The result is uncomputed automatically, so it cannot be measured
        directly: use it as a control, or add it into a clean register with
        ``+=``.
        """
        if not isinstance(other, (int, float, Quant)):
            return NotImplemented

        result = self.copy([other])

        def inner_add(result):
            result += other

        return Qreal(undo(inner_add, result), exp=self.exp)

    def __sub__(self, other: Any):
        """Subtraction: ``self - other``, into a new auxiliary register.

        See :meth:`__add__`.
        """
        if not isinstance(other, (int, float, Quant)):
            return NotImplemented

        result = self.copy([other])

        def inner_sub(result):
            result -= other

        return Qreal(undo(inner_sub, result), exp=self.exp)

    def mul(self, other, result):
        """Quantum multiplication: accumulate ``self * other`` into ``result``.

        Performs an in-place multiply-accumulate: each bit of ``self``
        (the multiplier) controls an addition of the appropriately shifted
        ``other`` (the multiplicand) into ``result``.

        This method is used internally by :meth:`~ket.qint.Qreal.__mul__`.

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
            ctrl(q, _addi_signed)(result, other, m=2 ** (i + m))

    def __mul__(self, other: Any):
        """Multiplication: ``self * other``, into a new auxiliary register.

        The result has as many qubits as ``self``, and its exponent is the sum
        of the exponents of the operands. See :meth:`__add__`.
        """
        if not isinstance(other, (int, float, Quant)):
            return NotImplemented

        result_quant = self.ket_process.alloc_aux(len(self), depends_on=[self, other])
        new_exp = self.exp + other.exp if isinstance(other, Qreal) else self.exp

        raw_result = undo(partial(self.mul, other), result_quant)
        return Qreal(raw_result, exp=new_exp)

    def __truediv__(self, other: Any):
        """Division: ``self / other``, into a new auxiliary register.

        ``other`` can be another :class:`Qint`/:class:`Qreal` or a number.
        The result keeps the size and exponent of ``self``, and is truncated
        toward zero (``-7 / 2`` is ``-3`` for a :class:`Qint`). Values that do
        not fit wrap around (two's complement). Dividing by zero gives an
        undefined result for a quantum divisor, and raises
        :class:`ZeroDivisionError` for a number.

        A number that is not an integer is rounded to ``len(self)``
        fractional bits. The division uses about :math:`2(n + m)` auxiliary
        qubits, for an :math:`n`-qubit ``self`` and an :math:`m`-qubit
        ``other``. See :meth:`__add__` for how to use the result.
        """
        if isinstance(other, Quant) and not isinstance(other, Qreal):
            return NotImplemented
        if not isinstance(other, (int, float, Qreal)):
            return NotImplemented

        n = len(self)
        if isinstance(other, Qreal):
            # x / y = (a / 2**ea) / (b / 2**eb) = (a * 2**eb / b) / 2**ea
            shift = other.exp
            divisor = other
            negative = False
        else:
            if other == 0:
                raise ZeroDivisionError("division by zero")
            # x / c = (a * 2**f / round(|c| * 2**f)) / 2**ea
            shift = 0
            while shift < n and abs(other) * 2**shift != round(abs(other) * 2**shift):
                shift += 1
            divisor = round(abs(other) * 2**shift)
            negative = other < 0
            if divisor == 0:
                raise ZeroDivisionError(
                    f"{other} is zero at the precision of the dividend"
                )

        size = _division_size(n, divisor, shift)
        workspace = self.ket_process.alloc_aux(
            size, depends_on=[self, other] if isinstance(other, Quant) else [self]
        )
        result = undo(
            partial(
                _divide,
                dividend=self,
                divisor=divisor,
                shift=shift,
                divisor_negative=negative,
            ),
            workspace,
        )
        num_q = n + max(shift, 0)
        return Qreal(result[num_q - n : num_q], exp=self.exp)

    def __neg__(self):
        """Negation: ``-self``, into a new auxiliary register.

        See :meth:`__add__`.
        """
        def neg(qint):
            X(qint)
            qint += 1

        result = self.copy()
        return Qreal(undo(neg, result), exp=self.exp)

    def __lt__(self, other: Any):
        """Comparison ``self < other``: one auxiliary qubit, in state 1 when true.

        The qubit is uncomputed automatically, so it cannot be measured
        directly: use it as a control, for example ``with control(a < b):``.
        """
        if not isinstance(other, (int, float, Quant)):
            return NotImplemented

        result = self.copy()

        def lt(result):
            result -= other

        return undo(lt, result)[0]

    def __gt__(self, other: Any):
        """Comparison ``self > other``: one auxiliary qubit, in state 1 when true.

        The qubit is uncomputed automatically, so it cannot be measured
        directly: use it as a control, for example ``with control(a > b):``.
        """
        if not isinstance(other, (int, float, Quant)):
            return NotImplemented

        result = -self + other
        return result[0]

    def __ge__(self, other: Any):
        """Comparison ``self >= other``: one auxiliary qubit, in state 1 when true.

        The qubit is uncomputed automatically, so it cannot be measured
        directly: use it as a control, for example ``with control(a >= b):``.
        """
        if not isinstance(other, (int, float, Quant)):
            return NotImplemented

        return undo(X, (self < other))

    def __le__(self, other: Any):
        """Comparison ``self <= other``: one auxiliary qubit, in state 1 when true.

        The qubit is uncomputed automatically, so it cannot be measured
        directly: use it as a control, for example ``with control(a <= b):``.
        """
        if not isinstance(other, (int, float, Quant)):
            return NotImplemented

        return undo(X, (self > other))

    def __eq__(self, other: Any):
        """Comparison ``self == other``: one auxiliary qubit, in state 1 when true.

        The qubit is uncomputed automatically, so it cannot be measured
        directly: use it as a control, for example ``with control(a == b):``.
        """
        if not isinstance(other, (int, float, Quant)):
            return NotImplemented

        cond = self - other
        result = self.ket_process.alloc_aux(depends_on=[cond])

        def eq(result):
            with control(Quant.__eq__(cond, 0)):
                X(result)

        return undo(eq, result)

    def __ne__(self, other: Any):
        """Comparison ``self != other``: one auxiliary qubit, in state 1 when true.

        The qubit is uncomputed automatically, so it cannot be measured
        directly: use it as a control, for example ``with control(a != b):``.
        """
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
