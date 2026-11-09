"""QUBO modeling helpers.

Tools for writing QUBO (Quadratic Unconstrained Binary Optimization) problems
as Ket Hamiltonians. Each binary variable :math:`x_i` is a qubit, read through
the observable :func:`~ket.gates.B`, :math:`x_i = (1 - Z_i)/2`.

The module has two layers:

- **Functions** that build Hamiltonian pieces: integer variables in binary
  (:func:`binary`) or one-hot (:func:`one_hot`) encoding, and penalty terms for
  constraints (:func:`exactly_one`, :func:`at_most_one`, :func:`equal`,
  :func:`less_equal`, :func:`greater_equal`). They work on process qubits or on
  placeholder qubits from ``with obs(n) as q:``, like any other Ket observable.
- The :class:`QUBO` model. Its variables are :class:`Expression` objects:
  arithmetic builds new expressions and comparisons build :class:`Constraint`
  objects. ``qubo += expr`` adds to the objective and ``qubo += constraint``
  adds a restriction. The model allocates slack qubits, picks penalty weights,
  and compiles everything into one unbound :class:`~ket.expv.Hamiltonian`.

Example:

    .. code-block:: python

        from ket.qulib.qubo import QUBO
        from ket.qulib import exact_solver

        # Knapsack: pick items to maximize value with total weight <= 5
        values, weights = [3, 4, 5], [2, 3, 4]

        qubo = QUBO()
        x = qubo.binary_vars(3)
        qubo += -sum(v * xi for v, xi in zip(values, x))      # maximize value
        qubo += sum(w * xi for w, xi in zip(weights, x)) <= 5, "capacity"

        state, _ = exact_solver(qubo, qubo.num_qubits)
        print(qubo.evaluate(state), qubo.is_feasible(state))   # -7.0 True
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access,too-many-lines

from collections import defaultdict
from collections.abc import Iterable
from functools import reduce
from itertools import combinations
from math import floor, isclose, prod
from numbers import Number
import operator
from typing import Literal

from ..base import Quant
from ..expv import Hamiltonian, ObsQubits, Pauli
from ..gates import B

__all__ = [
    "binary",
    "one_hot",
    "slack_size",
    "exactly_one",
    "at_most_one",
    "equal",
    "less_equal",
    "greater_equal",
    "Expression",
    "Constraint",
    "QUBO",
]

Qubits = Quant | ObsQubits
Expr = Hamiltonian | Pauli | Number
Encoding = Literal["binary", "one_hot"]

_TOL = 1e-9


def _as_qubits(qubits) -> Qubits:
    if isinstance(qubits, (Quant, ObsQubits)):
        return qubits
    return reduce(operator.add, qubits)


def _zero(qubits: Qubits) -> Hamiltonian:
    return Hamiltonian([], qubits.ket_process)


def _check_encoding(encoding: str):
    if encoding not in ("binary", "one_hot"):
        raise ValueError(
            f"Unknown encoding {encoding!r}; expected 'binary' or 'one_hot'"
        )


def slack_size(max_value: int, encoding: Encoding = "binary") -> int:
    """Number of qubits needed to encode an integer in ``[0, max_value]``.

    Args:
        max_value: Largest value the variable must reach.
        encoding: ``"binary"`` needs ``max_value.bit_length()`` qubits;
            ``"one_hot"`` needs ``max_value + 1`` qubits.
    """
    _check_encoding(encoding)
    max_value = int(max_value)
    if max_value < 0:
        raise ValueError("max_value must be non-negative")
    if encoding == "binary":
        return max_value.bit_length()
    return max_value + 1


def _binary_weights(num_qubits: int, max_value: int | None) -> list[int]:
    if num_qubits == 0:
        raise ValueError("A binary variable needs at least one qubit")
    if max_value is None:
        max_value = (1 << num_qubits) - 1
    if slack_size(max_value, "binary") != num_qubits:
        raise ValueError(
            f"max_value={max_value} needs {slack_size(max_value)} qubits "
            f"in binary encoding, but got {num_qubits}"
        )
    weights = [1 << k for k in range(num_qubits - 1)]
    weights.append(max_value - sum(weights))
    return weights


def binary(qubits: Qubits, max_value: int | None = None) -> Hamiltonian:
    r"""Integer variable in bounded binary encoding.

    Qubit :math:`k` has weight :math:`2^k`, except the last one, whose weight
    completes exactly ``max_value``. The variable then takes every value from
    ``0`` to ``max_value`` and none above. For example, ``max_value=5`` uses 3
    qubits with weights 1, 2 and 2.

    .. math::

        v = \sum_{k=0}^{n-2} 2^k x_k + \left(m - 2^{n-1} + 1\right) x_{n-1}

    Example:

        .. code-block:: python

            from ket import *
            from ket.qulib.qubo import binary

            with obs(3) as q:
                v = binary(q, max_value=5)    # 1*B(q0) + 2*B(q1) + 2*B(q2)

    Args:
        qubits: Qubits holding the variable. Must have exactly
            :func:`slack_size(max_value) <slack_size>` qubits.
        max_value: Largest value of the variable. If ``None``, it is
            :math:`2^n - 1` (plain binary).

    Returns:
        Observable whose eigenvalue is the encoded integer.
    """
    qubits = _as_qubits(qubits)
    weights = _binary_weights(len(qubits), max_value)
    return sum(w * B(q) for w, q in zip(weights, qubits))


def one_hot(qubits: Qubits, values: Iterable[float] | None = None) -> Hamiltonian:
    r"""Variable in one-hot encoding.

    .. math::

        v = \sum_k v_k x_k

    The variable equals ``values[k]`` when only qubit ``k`` is set. This
    observable alone does not force a single qubit to be set: add the
    :func:`exactly_one` penalty for that (:meth:`QUBO.one_hot_var` does it for
    you).

    Args:
        qubits: Qubits holding the variable, one per value.
        values: Value of each qubit. Defaults to ``0, 1, ..., n - 1``.

    Returns:
        Observable whose eigenvalue is the encoded value.
    """
    qubits = _as_qubits(qubits)
    values = range(len(qubits)) if values is None else list(values)
    if len(values) != len(qubits):
        raise ValueError(
            f"Got {len(values)} values for a {len(qubits)}-qubit one-hot variable"
        )
    return sum(v * B(q) for v, q in zip(values, qubits))


def exactly_one(qubits: Qubits) -> Hamiltonian:
    r"""Penalty for the one-hot restriction: exactly one qubit is set.

    .. math::

        \left(\sum_k x_k - 1\right)^2

    Zero when exactly one qubit is :math:`\left|1\right>`, at least 1 otherwise.
    """
    qubits = _as_qubits(qubits)
    return (sum(B(q) for q in qubits) - 1) ** 2


def at_most_one(qubits: Qubits) -> Hamiltonian:
    r"""Penalty for the restriction: at most one qubit is set.

    .. math::

        \sum_{i < j} x_i x_j

    Zero when no more than one qubit is :math:`\left|1\right>`, at least 1
    otherwise.
    """
    qubits = _as_qubits(qubits)
    if len(qubits) < 2:
        return _zero(qubits)
    return sum(B(a) * B(b) for a, b in combinations(qubits, 2))


def equal(expr: Expr, value: float) -> Hamiltonian:
    r"""Penalty for the restriction ``expr == value``.

    .. math::

        \left(\text{expr} - \text{value}\right)^2

    Args:
        expr: Diagonal observable, usually a weighted sum of :func:`~ket.gates.B`.
        value: Target value.
    """
    return (expr - value) ** 2


def _slack_value(slack: Qubits, encoding: str) -> tuple[Hamiltonian, Hamiltonian]:
    _check_encoding(encoding)
    if encoding == "binary":
        return binary(slack), None
    return one_hot(slack), exactly_one(slack)


def less_equal(
    expr: Expr, bound: float, slack: Qubits, encoding: Encoding = "binary"
) -> Hamiltonian:
    r"""Penalty for the restriction ``expr <= bound``.

    Adds an integer slack variable :math:`s \ge 0` stored in ``slack``:

    .. math::

        \left(\text{expr} + s - \text{bound}\right)^2

    With ``encoding="one_hot"`` the :func:`exactly_one` penalty on the slack
    qubits is added too. The slack must reach ``bound - min(expr)`` for every
    feasible point to have zero penalty; use :func:`slack_size` to size it.
    Extra slack range is harmless. The restriction is exact for integer-valued
    ``expr`` and ``bound``.

    Args:
        expr: Diagonal observable.
        bound: Upper bound.
        slack: Qubits for the slack variable (not used anywhere else).
        encoding: Slack encoding, ``"binary"`` or ``"one_hot"``.
    """
    s, s_penalty = _slack_value(slack, encoding)
    penalty = equal(expr + s, bound)
    return penalty if s_penalty is None else penalty + s_penalty


def greater_equal(
    expr: Expr, bound: float, slack: Qubits, encoding: Encoding = "binary"
) -> Hamiltonian:
    r"""Penalty for the restriction ``expr >= bound``.

    Same as :func:`less_equal`, with

    .. math::

        \left(\text{expr} - s - \text{bound}\right)^2

    The slack must reach ``max(expr) - bound``.
    """
    s, s_penalty = _slack_value(slack, encoding)
    penalty = equal(expr - s, bound)
    return penalty if s_penalty is None else penalty + s_penalty


def _as_hamiltonian(expr: Expr) -> Hamiltonian:
    if isinstance(expr, Hamiltonian):
        return expr
    if isinstance(expr, Pauli):
        return Hamiltonian([expr], expr.ket_process)
    if isinstance(expr, Number):
        return Hamiltonian([Pauli(None, None, _map={}, _coef=expr)], None)
    raise TypeError(f"Expected a Hamiltonian, Pauli or number, got {type(expr)}")


def _diagonal_terms(expr: Expr) -> list[tuple[float, list[int]]]:
    """Return ``(coef, z_qubits)`` for each term, checking the expr is diagonal."""
    terms = []
    for term in _as_hamiltonian(expr).terms:
        qubits = [q for q, p in term.map.items() if p != "I"]
        if any(term.map[q] != "Z" for q in qubits):
            raise ValueError("QUBO expressions must only contain Z and I operators")
        coef = complex(term.coef)
        if not isclose(coef.imag, 0.0, abs_tol=_TOL):
            raise ValueError("QUBO expressions must have real coefficients")
        terms.append((coef.real, qubits))
    return terms


def _binary_poly(expr: Expr) -> dict[frozenset[int], float]:
    """Expand ``expr`` into a polynomial over the binary variables."""
    poly = defaultdict(float)
    for coef, qubits in _diagonal_terms(expr):
        # Z = 1 - 2x
        for r in range(len(qubits) + 1):
            for subset in combinations(qubits, r):
                poly[frozenset(subset)] += coef * (-2) ** r
    return poly


def _bounds(expr: Expr) -> tuple[float, float]:
    """Cheap lower and upper bounds of ``expr`` over all bit assignments."""
    poly = _binary_poly(expr)
    const = poly.pop(frozenset(), 0.0)
    low = const + sum(c for c in poly.values() if c < 0)
    high = const + sum(c for c in poly.values() if c > 0)
    return low, high


def _evaluate(expr: Expr, bits: list[int]) -> float:
    return sum(
        coef * prod(1 - 2 * bits[q] for q in qubits)
        for coef, qubits in _diagonal_terms(expr)
    )


def _is_integral(value: float) -> bool:
    return isclose(value, round(value), abs_tol=_TOL)


def _poly_str(poly: dict[frozenset[int], float]) -> str:
    terms = sorted(
        ((sorted(k), c) for k, c in poly.items() if abs(c) > _TOL),
        key=lambda kc: (len(kc[0]), kc[0]),
    )
    # Constant last, like a polynomial written by hand.
    terms = [t for t in terms if t[0]] + [t for t in terms if not t[0]]
    out = ""
    for qubits, coef in terms:
        sign = "-" if coef < 0 else "+"
        mag = abs(coef)
        names = "*".join(f"x{q}" for q in qubits)
        if not names:
            body = f"{mag:g}"
        elif isclose(mag, 1.0):
            body = names
        else:
            body = f"{mag:g}*{names}"
        out += f" {sign} {body}" if out else ("-" if sign == "-" else "") + body
    return out or "0"


class Expression(Hamiltonian):
    r"""A diagonal expression over the binary variables of a :class:`QUBO`.

    Expressions come from :meth:`QUBO.binary_var`, :meth:`QUBO.binary_vars`,
    :meth:`QUBO.int_var` and :meth:`QUBO.one_hot_var`, or from wrapping an
    unbound diagonal Hamiltonian, ``Expression(B(q))``. They support:

    - ``+``, ``-``, ``*``, ``/``, ``**`` and unary ``-``, with numbers, other
      expressions or unbound diagonal Hamiltonians. The product follows the
      Pauli algebra, so :math:`x \cdot x = x` for a binary variable.
    - ``==``, ``<=``, ``>=``, ``<`` and ``>``, which return a
      :class:`Constraint` to add to the model with ``qubo += constraint``.
      Strict comparisons assume integer values: ``e < b`` is ``e <= b - 1``.

    ``!=`` is not supported, and expressions are not hashable.

    An expression is an unbound :class:`~ket.expv.Hamiltonian`, so it can also
    be bound to qubits or passed to :func:`~ket.operations.exp_value`.

    Example:

        .. code-block:: python

            qubo = QUBO()
            x, y = qubo.binary_vars(2)
            n = qubo.int_var(5)

            cost = 3 * x + 2 * y - x * y     # Expression
            cap = n + 2 * x <= 6             # Constraint

    Args:
        expr: Unbound diagonal Hamiltonian, Pauli or number.
        qubits: Qubits of the variable, when the expression is one.
    """

    def __init__(self, expr: Expr = 0, qubits: ObsQubits | None = None):
        h = _as_hamiltonian(expr)
        if h.is_bound:
            raise ValueError(
                "QUBO expressions must use placeholder qubits, not process qubits"
            )
        _diagonal_terms(h)
        super().__init__(list(h.terms), None, _index=h._index)
        self.qubits = qubits
        """Qubits of the variable, or ``None`` for a compound expression."""

    @classmethod
    def _wrap(cls, h: Hamiltonian) -> Expression:
        """Wrap the result of an operation on checked operands, without re-checking it."""
        result = cls.__new__(cls)
        Hamiltonian.__init__(result, h.terms, None, _index=h._index)
        result.qubits = None
        return result

    @staticmethod
    def _operand(other) -> Hamiltonian | None:
        """``other`` as a Hamiltonian checked for use in an expression, or ``None``."""
        if isinstance(other, Expression):
            return other
        if isinstance(other, (Hamiltonian, Pauli, Number)):
            h = _as_hamiltonian(other)
            if h.is_bound:
                raise ValueError(
                    "QUBO expressions must use placeholder qubits, not process qubits"
                )
            _diagonal_terms(h)
            return h
        return None

    def __add__(self, other) -> Expression:
        other = self._operand(other)
        if other is None:
            return NotImplemented
        return Expression._wrap(Hamiltonian.__add__(self, other))

    __radd__ = __add__

    def __sub__(self, other) -> Expression:
        other = self._operand(other)
        if other is None:
            return NotImplemented
        return self + Hamiltonian.__mul__(other, -1.0)

    def __rsub__(self, other) -> Expression:
        return -self + other

    def __mul__(self, other) -> Expression:
        if isinstance(other, Number):
            return Expression._wrap(Hamiltonian.__mul__(self, other))
        other = self._operand(other)
        if other is None:
            return NotImplemented
        return Expression._wrap(Hamiltonian.__matmul__(self, other))

    __rmul__ = __mul__
    __matmul__ = __mul__
    __rmatmul__ = __mul__

    def __truediv__(self, other) -> Expression:
        if not isinstance(other, Number):
            return NotImplemented
        return self * (1.0 / other)

    def __neg__(self) -> Expression:
        return self * -1.0

    def __pos__(self) -> Expression:
        return self

    def __pow__(self, exp: int) -> Expression:
        if not isinstance(exp, int) or exp < 0:
            raise ValueError("Only non-negative integer powers are supported")
        result = Expression(1)
        for _ in range(exp):
            result = result * self
        return result

    def _compare(self, other, sense: str) -> Constraint:
        other = self._operand(other)
        if other is None:
            return NotImplemented
        return Constraint(self - other, sense)

    def __eq__(self, other) -> Constraint:  # type: ignore[override]
        return self._compare(other, "==")

    def __le__(self, other) -> Constraint:
        return self._compare(other, "<=")

    def __ge__(self, other) -> Constraint:
        return self._compare(other, ">=")

    def __lt__(self, other) -> Constraint:
        constraint = self._compare(other, "<=")
        if constraint is NotImplemented:
            return constraint
        constraint._strict(-1)
        return constraint

    def __gt__(self, other) -> Constraint:
        constraint = self._compare(other, ">=")
        if constraint is NotImplemented:
            return constraint
        constraint._strict(+1)
        return constraint

    def __ne__(self, other):  # type: ignore[override]
        raise TypeError("'!=' constraints are not supported in a QUBO")

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"<Ket 'Expression' {_poly_str(_binary_poly(self))}>"


class Constraint:
    """A restriction on :class:`Expression` objects, written ``lhs sense rhs``.

    Built by comparing expressions (``==``, ``<=``, ``>=``, ``<``, ``>``) and
    added to a model with ``qubo += constraint``. To set a penalty weight or a
    name, add a tuple, ``qubo += constraint, 10.0, "name"``, or call
    :meth:`QUBO.add`.

    The constant part of the comparison is moved to the right-hand side, so
    ``x + 1 <= y`` becomes ``x - y <= -1``.

    .. note::
        A constraint has no truth value: ``if x == y:`` raises
        :class:`TypeError`.
    """

    def __init__(self, expr: Expression, sense: str):
        poly = _binary_poly(expr)
        constant = poly.get(frozenset(), 0.0)
        self.lhs: Expression = expr - constant
        """Left-hand side, without a constant term."""
        self.sense: str = sense
        """``"=="``, ``"<="`` or ``">="``."""
        self.rhs: float = -constant
        """Right-hand side constant."""

    def _strict(self, shift: int):
        poly = _binary_poly(self.lhs)
        if not (_is_integral(self.rhs) and all(map(_is_integral, poly.values()))):
            raise ValueError(
                "Strict comparisons need integer coefficients; use <= or >= instead"
            )
        self.rhs += shift

    def __bool__(self):
        raise TypeError(
            "A QUBO constraint has no truth value; add it to the model with "
            "'qubo += constraint'"
        )

    def __repr__(self) -> str:
        return (
            f"<Ket 'Constraint' {_poly_str(_binary_poly(self.lhs))} "
            f"{self.sense} {self.rhs:g}>"
        )


class _Constraint:  # pylint: disable=too-few-public-methods
    def __init__(self, name: str, penalty: Hamiltonian, weight: float | None):
        self.name = name
        self.penalty = penalty
        self.weight = weight


class QUBO:  # pylint: disable=too-many-public-methods
    r"""A QUBO problem with constraints, compiled to a Hamiltonian.

    The problem is built on placeholder qubits (:class:`~ket.expv.ObsQubits`),
    so it does not depend on a process. Create variables with
    :meth:`binary_var`, :meth:`binary_vars`, :meth:`int_var` or
    :meth:`one_hot_var`; they are :class:`Expression` objects, so the model is
    written with plain operators:

    - ``qubo += expr`` adds ``expr`` to the objective (minimized) and
      ``qubo -= expr`` subtracts it (to maximize).
    - ``qubo += expr <= bound`` (or ``==``, ``>=``, ``<``, ``>``) adds a
      constraint. Add a tuple to set a penalty weight and/or a name,
      ``qubo += (expr <= bound, 10.0, "capacity")``, or call :meth:`add`.
    - ``qubo += [c1, c2, ...]`` adds every item of a list.

    The methods :meth:`minimize`, :meth:`maximize`, :meth:`equal`,
    :meth:`less_equal`, :meth:`greater_equal`, :meth:`exactly_one`,
    :meth:`at_most_one` and :meth:`add_penalty` do the same with explicit
    calls, and also accept raw qubits from :meth:`alloc`. Inequalities get
    their slack qubits allocated automatically.

    :meth:`hamiltonian` compiles the problem into

    .. math::

        H = \text{objective} + \sum_c A_c \, P_c

    where :math:`P_c \ge 0` is the penalty of constraint :math:`c` (zero only
    when it holds) and :math:`A_c` its weight. The result is an unbound
    :class:`~ket.expv.Hamiltonian`. Calling the QUBO, ``qubo(qubits)``, binds it,
    so a :class:`QUBO` can be passed anywhere a
    ``Callable[[Quant], Hamiltonian]`` is expected, such as
    :func:`~ket.qulib.exact_solver` or :func:`~ket.qulib.energy`.

    If a constraint has no weight, it gets :math:`\Delta + 1`, where
    :math:`\Delta` is an upper bound on the objective range (max minus min).
    For integer-valued constraints, any violation costs at least 1, so every
    infeasible state has higher energy than every feasible one.

    Example:

        .. code-block:: python

            from ket import *
            from ket.qulib.qubo import QUBO

            # Assign 3 tasks to 2 machines, each task to exactly one machine,
            # and machine 0 takes at most 1 task.
            cost = [[1, 3], [2, 1], [4, 2]]

            qubo = QUBO()
            x = [qubo.binary_vars(2) for _ in range(3)]    # x[t][m]
            qubo += sum(cost[t][m] * x[t][m] for t in range(3) for m in range(2))
            qubo += [(sum(x[t]) == 1, f"task {t}") for t in range(3)]
            qubo += sum(x[t][0] for t in range(3)) <= 1, "machine 0"

            p = Process()
            q = p.alloc(qubo.num_qubits)
            ev = exp_value(qubo(q))

    Args:
        num_qubits: Number of decision qubits to allocate at once; they are
            available from :attr:`qubits`.
    """

    def __init__(self, num_qubits: int = 0):
        self._num_qubits = 0
        self._objective: Hamiltonian = _as_hamiltonian(0)
        self._constraints: list[_Constraint] = []
        self.qubits: ObsQubits = self.alloc(num_qubits)
        """Decision qubits allocated by the constructor."""

    @property
    def num_qubits(self) -> int:
        """Total number of qubits, including slack qubits."""
        return self._num_qubits

    @property
    def objective(self) -> Expression:
        """The objective, without penalties."""
        return Expression(self._objective)

    @property
    def constraints(self) -> list[tuple[str, Hamiltonian, float | None]]:
        """``(name, penalty, weight)`` of each constraint. ``weight`` is ``None``
        when it is chosen automatically."""
        return [(c.name, c.penalty, c.weight) for c in self._constraints]

    def alloc(self, num_qubits: int = 1) -> ObsQubits:
        """Allocate new placeholder qubits for decision variables.

        Args:
            num_qubits: Number of qubits.

        Returns:
            The new qubits.
        """
        if num_qubits < 0:
            raise ValueError("num_qubits must be non-negative")
        start = self._num_qubits
        self._num_qubits += num_qubits
        return ObsQubits(_qubits=list(range(start, self._num_qubits)))

    def binary_var(self) -> Expression:
        """Allocate one binary (0/1) variable."""
        qubits = self.alloc(1)
        return Expression(B(qubits), qubits)

    def binary_vars(self, num_vars: int) -> list[Expression]:
        """Allocate ``num_vars`` binary (0/1) variables."""
        return [self.binary_var() for _ in range(num_vars)]

    def int_var(
        self,
        max_value: int,
        encoding: Encoding = "binary",
        penalty: float | None = None,
        name: str | None = None,
    ) -> Expression:
        """Allocate an integer variable in ``[0, max_value]``.

        With ``encoding="binary"`` it uses :func:`binary`: few qubits and no
        constraint, since every bit assignment is a valid value. With
        ``encoding="one_hot"`` it is :meth:`one_hot_var` over
        ``0, 1, ..., max_value``.

        Args:
            max_value: Largest value, at least 1.
            encoding: ``"binary"`` or ``"one_hot"``.
            penalty: Weight of the one-hot constraint (one-hot only).
            name: Name of the one-hot constraint (one-hot only).
        """
        _check_encoding(encoding)
        if max_value < 1:
            raise ValueError("max_value must be at least 1")
        if encoding == "one_hot":
            return self.one_hot_var(max_value + 1, penalty, name)
        qubits = self.alloc(slack_size(max_value, "binary"))
        return Expression(binary(qubits, max_value), qubits)

    def one_hot_var(
        self,
        values: int | Iterable[float],
        penalty: float | None = None,
        name: str | None = None,
    ) -> Expression:
        """Allocate a variable in one-hot encoding and add its
        :meth:`exactly_one` constraint.

        Args:
            values: The values the variable can take, or an integer ``n`` for
                ``0, 1, ..., n - 1``.
            penalty: Weight of the one-hot constraint.
            name: Name of the one-hot constraint.
        """
        values = list(range(values)) if isinstance(values, int) else list(values)
        qubits = self.alloc(len(values))
        self.exactly_one(qubits, penalty=penalty, name=name)
        return Expression(one_hot(qubits, values), qubits)

    def _check(self, expr: Expr) -> Hamiltonian:
        expr = _as_hamiltonian(expr)
        if expr.is_bound:
            raise ValueError(
                "QUBO expressions must use the QUBO placeholder qubits, "
                "not qubits from a process"
            )
        if expr.num_qubits > self._num_qubits:
            raise ValueError(
                f"Expression uses qubit {expr.num_qubits - 1}, "
                f"but the QUBO has only {self._num_qubits} qubits"
            )
        _diagonal_terms(expr)
        return Hamiltonian(list(expr.terms), None)

    def minimize(self, expr: Expr) -> QUBO:
        """Add ``expr`` to the objective. Can be called more than once."""
        self._objective = self._objective + self._check(expr)
        return self

    def maximize(self, expr: Expr) -> QUBO:
        """Add ``-expr`` to the objective. Can be called more than once."""
        return self.minimize(-_as_hamiltonian(expr))

    def add_penalty(
        self, penalty: Expr, weight: float | None = None, name: str | None = None
    ) -> QUBO:
        """Add a custom constraint given as a penalty Hamiltonian.

        Args:
            penalty: Diagonal observable that is zero on feasible states and
                positive (at least 1, for the automatic weight) otherwise.
            weight: Penalty weight. If ``None``, it is chosen at compile time.
            name: Name used by :meth:`violations`.
        """
        penalty = self._check(penalty)
        if name is None:
            name = f"constraint {len(self._constraints)}"
        self._constraints.append(_Constraint(name, penalty, weight))
        return self

    def exactly_one(
        self, qubits: Qubits, penalty: float | None = None, name: str | None = None
    ) -> QUBO:
        """Restrict ``qubits`` to one-hot: exactly one is set. See :func:`exactly_one`."""
        return self.add_penalty(exactly_one(qubits), penalty, name)

    def at_most_one(
        self, qubits: Qubits, penalty: float | None = None, name: str | None = None
    ) -> QUBO:
        """Restrict ``qubits`` so at most one is set. See :func:`at_most_one`."""
        return self.add_penalty(at_most_one(qubits), penalty, name)

    def equal(
        self,
        expr: Expr,
        value: float,
        penalty: float | None = None,
        name: str | None = None,
    ) -> QUBO:
        """Restrict ``expr == value``. See :func:`equal`."""
        return self.add_penalty(equal(self._check(expr), value), penalty, name)

    def _inequality(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, expr, bound, penalty, name, encoding, sign
    ) -> QUBO:
        _check_encoding(encoding)
        expr = self._check(expr)
        low, high = _bounds(expr)
        # Range the slack must cover so every feasible state reaches zero penalty.
        if sign > 0:  # expr <= bound
            if high <= bound + _TOL:
                return self  # always satisfied
            slack_max = bound - low
        else:  # expr >= bound
            if low >= bound - _TOL:
                return self
            slack_max = high - bound
        if slack_max < -_TOL:
            raise ValueError(f"Constraint {name or ''} can never be satisfied")

        slack_max = floor(slack_max + _TOL)
        if slack_max == 0:
            return self.equal(expr, bound, penalty, name)

        slack = self.alloc(slack_size(slack_max, encoding))
        if encoding == "binary":
            s = binary(slack, slack_max)
            restriction = equal(expr + sign * s, bound)
        else:
            restriction = equal(expr + sign * one_hot(slack), bound) + exactly_one(
                slack
            )
        return self.add_penalty(restriction, penalty, name)

    def less_equal(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        expr: Expr,
        bound: float,
        penalty: float | None = None,
        name: str | None = None,
        encoding: Encoding = "binary",
    ) -> QUBO:
        """Restrict ``expr <= bound``, allocating slack qubits as needed.

        The slack covers exactly ``[0, bound - min(expr)]``, using a lower
        bound of ``expr``. No qubit is allocated when the restriction always
        holds. See :func:`less_equal`.

        Args:
            expr: Diagonal observable on the QUBO qubits, integer-valued.
            bound: Upper bound.
            penalty: Constraint weight. If ``None``, chosen at compile time.
            name: Name used by :meth:`violations`.
            encoding: Slack encoding, ``"binary"`` (fewer qubits) or
                ``"one_hot"``.
        """
        return self._inequality(expr, bound, penalty, name, encoding, +1)

    def greater_equal(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        expr: Expr,
        bound: float,
        penalty: float | None = None,
        name: str | None = None,
        encoding: Encoding = "binary",
    ) -> QUBO:
        """Restrict ``expr >= bound``, allocating slack qubits as needed.

        Same as :meth:`less_equal`; the slack covers ``[0, max(expr) - bound]``.
        """
        return self._inequality(expr, bound, penalty, name, encoding, -1)

    def add(
        self,
        constraint: Constraint,
        penalty: float | None = None,
        name: str | None = None,
        encoding: Encoding = "binary",
    ) -> QUBO:
        """Add a :class:`Constraint` built with ``==``, ``<=``, ``>=``, ``<`` or ``>``.

        ``qubo += constraint`` is a shorthand for this method.

        Args:
            constraint: The constraint, e.g. ``x + y <= 1``.
            penalty: Constraint weight. If ``None``, chosen at compile time.
            name: Name used by :meth:`violations`.
            encoding: Slack encoding for inequalities, ``"binary"`` or
                ``"one_hot"``.
        """
        if not isinstance(constraint, Constraint):
            raise TypeError(f"Expected a Constraint, got {type(constraint)}")
        lhs, rhs = constraint.lhs, constraint.rhs
        if constraint.sense == "==":
            return self.equal(lhs, rhs, penalty, name)
        if constraint.sense == "<=":
            return self.less_equal(lhs, rhs, penalty, name, encoding)
        return self.greater_equal(lhs, rhs, penalty, name, encoding)

    def __iadd__(self, other) -> QUBO:
        if isinstance(other, Constraint):
            return self.add(other)
        if isinstance(other, tuple):
            if not other or not isinstance(other[0], Constraint):
                raise TypeError("A tuple added to a QUBO must start with a Constraint")
            penalty = name = None
            for option in other[1:]:
                if isinstance(option, str):
                    name = option
                elif isinstance(option, Number):
                    penalty = option
                else:
                    raise TypeError(
                        "Constraint options must be a penalty (number) or a name (str)"
                    )
            return self.add(other[0], penalty, name)
        if isinstance(other, list):
            for item in other:
                self.__iadd__(item)
            return self
        if isinstance(other, (Hamiltonian, Pauli, Number)):
            return self.minimize(other)
        return NotImplemented

    def __isub__(self, other) -> QUBO:
        if isinstance(other, (Hamiltonian, Pauli, Number)):
            return self.maximize(other)
        return NotImplemented

    def default_penalty(self) -> float:
        """Weight given to constraints without an explicit one."""
        low, high = _bounds(self._objective)
        return high - low + 1.0

    def hamiltonian(self) -> Hamiltonian:
        """Compile the objective and penalties into an unbound Hamiltonian."""
        default = self.default_penalty()
        result = self._objective
        for constraint in self._constraints:
            weight = default if constraint.weight is None else constraint.weight
            result = result + weight * constraint.penalty
        return result

    def bind(self, qubits: Quant) -> Hamiltonian:
        """Compile and bind the Hamiltonian to process qubits.

        ``qubo(qubits)`` is a shorthand for this method.

        Args:
            qubits: At least :attr:`num_qubits` qubits from a process.
        """
        return self.hamiltonian().bind(qubits)

    __call__ = bind

    def _bits(self, state: int | str | Iterable[int]) -> list[int]:
        n = self._num_qubits
        if isinstance(state, int):
            if not 0 <= state < (1 << n):
                raise ValueError(f"State {state} does not fit in {n} qubits")
            return [int(b) for b in f"{state:0{n}b}"] if n else []
        bits = [int(b) for b in state]
        if len(bits) != n or any(b not in (0, 1) for b in bits):
            raise ValueError(f"State must have {n} bits, all 0 or 1")
        return bits

    def evaluate(
        self, state: int | str | Iterable[int], expr: Expr | None = None
    ) -> float:
        """Classical value of an expression on a basis state.

        Use it to read the objective of a solution or decode a variable, e.g.
        ``qubo.evaluate(state, n)`` for ``n = qubo.int_var(5)``.

        Args:
            state: Basis state as an integer (qubit 0 is the most significant
                bit, as in :func:`~ket.qulib.energy`), a bit string, or a list of bits.
            expr: Expression to evaluate. Defaults to the objective.
        """
        bits = self._bits(state)
        return _evaluate(self._objective if expr is None else expr, bits)

    def energy(self, state: int | str | Iterable[int]) -> float:
        """Value of the compiled Hamiltonian (objective plus penalties) on a basis state."""
        return _evaluate(self.hamiltonian(), self._bits(state))

    def violations(self, state: int | str | Iterable[int]) -> list[str]:
        """Names of the constraints a basis state breaks.

        A broken inequality may only mean its slack qubits are not set to the
        matching value.
        """
        bits = self._bits(state)
        return [c.name for c in self._constraints if _evaluate(c.penalty, bits) > _TOL]

    def is_feasible(self, state: int | str | Iterable[int]) -> bool:
        """``True`` if the basis state satisfies every constraint."""
        return not self.violations(state)

    def __repr__(self) -> str:
        return (
            f"<Ket 'QUBO' num_qubits={self._num_qubits}, "
            f"constraints={len(self._constraints)}>"
        )
