"""Expected value calculation utilities.

This module provides :class:`~ket.expv.Pauli`, :class:`~ket.expv.Hamiltonian`,
:class:`~ket.expv.ExpValue`, and the :func:`~ket.expv.commutator` helper for
constructing quantum observables and computing their expectation values.

The preferred workflow is to use the :func:`~ket.gates.obs` context manager
together with gate functions (:func:`~ket.gates.X`, :func:`~ket.gates.Y`,
:func:`~ket.gates.Z`) to build Hamiltonians in a symbolic, Dirac-like notation.
The :func:`~ket.operations.exp_value` function then wraps the resulting
:class:`~ket.expv.Hamiltonian` into an :class:`~ket.expv.ExpValue` handle.

Example:

    .. code-block:: python

        from ket import *
        p = Process()
        q = p.alloc(2)
        CNOT(H(q[0]), q[1])     # prepare Bell state
        with obs():
            # Heisenberg XX + YY + ZZ Hamiltonian
            h = X(q[0])*X(q[1]) + Y(q[0])*Y(q[1]) + Z(q[0])*Z(q[1])
        ev = exp_value(h)
        print(ev.get())          # 1.0 for the Bell state |Φ+⟩

Hamiltonians can also be built without a process. ``with obs(n) as q:``
yields ``n`` placeholder qubits (:class:`~ket.expv.ObsQubits`). The result is
an *unbound* Hamiltonian that you bind to real qubits later, either with
:meth:`~ket.expv.Hamiltonian.bind` (or the call shorthand ``h(qubits)``) or by
passing the qubits to :func:`~ket.operations.exp_value` or
:func:`~ket.gates.evolve`:

    .. code-block:: python

        from ket import *
        with obs(2) as q:
            h = X(q[0])*X(q[1]) + Y(q[0])*Y(q[1]) + Z(q[0])*Z(q[1])

        p = Process()
        r = p.alloc(2)
        CNOT(H(r[0]), r[1])
        print(exp_value(h, r).get())   # same as exp_value(h(r))
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=duplicate-code,protected-access
import json
import operator
from numbers import Number
from itertools import product
from functools import reduce
from typing import Literal
from fractions import Fraction

from .base import Parameter, Process, Quant
from .clib.libket import HasProcess

__all__ = [
    "ObsQubits",
    "Pauli",
    "Hamiltonian",
    "commutator",
    "ExpValue",
]


class ObsQubits:
    """Placeholder qubits for building Hamiltonians without a process.

    Placeholder qubits are numbered ``0`` to ``n - 1``. A Hamiltonian built on
    them is *unbound*: it supports all the usual algebra, but it must be bound
    to qubits from a :class:`~ket.base.Process` before computing an
    expectation value or evolving a state. Binding maps placeholder ``i`` to
    the ``i``-th qubit of the given register.

    .. note::
        You usually get these from ``with obs(n) as q:`` rather than by
        instantiating this class directly.

    Example:

        .. code-block:: python

            from ket import *
            with obs(3) as q:
                h = sum(Z(q[i]) * Z(q[i + 1]) for i in range(2))

            p = Process()
            r = p.alloc(3)
            ev = exp_value(h, r)

    Args:
        num_qubits: Number of placeholder qubits.
    """

    def __init__(self, num_qubits: int = 0, *, _qubits: list[int] | None = None):
        self.qubits = list(range(num_qubits)) if _qubits is None else _qubits

    @property
    def ket_process(self) -> None:
        """Always ``None``: placeholder qubits do not belong to a process."""
        return None

    def __getitem__(self, key) -> ObsQubits:
        qubits = self.qubits.__getitem__(key)
        return ObsQubits(_qubits=qubits if isinstance(qubits, list) else [qubits])

    def __iter__(self):
        for qubit in self.qubits:
            yield ObsQubits(_qubits=[qubit])

    def __reversed__(self) -> ObsQubits:
        return ObsQubits(_qubits=list(reversed(self.qubits)))

    def __len__(self) -> int:
        return len(self.qubits)

    def __add__(self, other: ObsQubits) -> ObsQubits:
        if not isinstance(other, ObsQubits):
            return NotImplemented
        if any(qubit in other.qubits for qubit in self.qubits):
            raise ValueError("Cannot concatenate qubits with overlapping indices")
        return ObsQubits(_qubits=self.qubits + other.qubits)

    def at(self, index: list[int]) -> ObsQubits:
        """Return the placeholder qubits at the given positions."""
        return ObsQubits(_qubits=[self.qubits[i] for i in index])

    def __repr__(self) -> str:
        return f"<Ket 'ObsQubits' {self.qubits}>"


def _check_same_process(lhs: HasProcess, rhs: HasProcess):
    if lhs.ket_process is rhs.ket_process:
        return
    if lhs.ket_process is None or rhs.ket_process is None:
        raise ValueError(
            "Cannot combine bound and unbound operators; "
            "bind the unbound one to qubits first"
        )
    raise ValueError("different Ket processes")


def _as_quant(qubits) -> Quant:
    if not isinstance(qubits, Quant):
        qubits = reduce(operator.add, qubits)
    if not isinstance(qubits, Quant):
        raise TypeError("Can only bind to qubits allocated from a Process")
    return qubits


def _pid(process) -> str:
    return "unbound" if process is None else f"pid={hex(id(process))}"


class Pauli(HasProcess):
    """Pauli operator for Hamiltonian creation.

    .. tip::
        The preferred way to create a Hamiltonian is by using the
        :func:`~ket.gates.obs` context manager, which allows you to define a Hamiltonian
        using a more intuitive syntax. However, you can also create Hamiltonians directly by
        instantiating this class.

    This class represents a Pauli operator for Hamiltonian creation. The primary usage of this class
    is to prepare a Hamiltonian by adding and multiplying Pauli operators and scalars for
    calculating the expected value of a quantum state.

    Args:
        pauli: Pauli operator type.
        qubits: Qubits to apply the Pauli operator to. Either qubits from a
            :class:`~ket.base.Process` or placeholder :class:`~ket.expv.ObsQubits`;
            the latter creates an unbound operator (see :meth:`bind`).
        process: *For internal usage*. Quantum process, default is the process of the given qubits.
        pauli_list: *For internal usage*. List of Pauli operators.
        qubits_list: *For internal usage*. List of Qubit.
        coef: *For internal usage*. Coefficient for the Pauli operator, default is 1.0.

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pauli: Literal["X", "Y", "Z", "I"] | None,
        qubits: Quant | ObsQubits | None,
        *,
        _process: Process | None = None,
        _map: dict[int, str] | None = None,
        _coef: float | complex | None = None,
    ):
        if qubits is not None and not isinstance(qubits, (Quant, ObsQubits)):
            qubits = reduce(operator.add, qubits)

        if pauli is not None and qubits is not None:
            super().__init__(ket_process=qubits.ket_process)
            self.map = {q: pauli for q in qubits.qubits}
            self.coef = 1.0
        else:
            assert (
                _map is not None and _coef is not None
            ), "If pauli and qubits are not provided, map and coef must be provided."
            super().__init__(ket_process=_process)
            self.map = _map
            self.coef = _coef

    def __neg__(self) -> Pauli:
        return -1.0 * self

    def __mul__(
        self, other: float | complex | Pauli | Parameter | Hamiltonian
    ) -> Pauli:

        if isinstance(other, (Number, Parameter)):
            return Pauli(
                None,
                None,
                _process=self.ket_process,
                _map=self.map,
                _coef=self.coef * other,
            )

        if isinstance(other, Hamiltonian):
            return other.__mul__(self)

        _check_same_process(self, other)

        if not set(self.map.keys()).isdisjoint(other.map.keys()):
            raise ValueError("Pauli operators act on the same qubits")

        return Pauli(
            None,
            None,
            _process=self.ket_process,
            _map={**self.map, **other.map},
            _coef=self.coef * other.coef,
        )

    def __matmul__(  # pylint:disable=too-many-branches
        self, rhs: Pauli | Hamiltonian
    ) -> Pauli:
        if isinstance(rhs, Hamiltonian):
            return Hamiltonian([self], self.ket_process) @ rhs

        _check_same_process(self, rhs)

        qubits = {*self.map.keys(), *rhs.map.keys()}
        result_map = {}
        coef = self.coef * rhs.coef

        for qubit in qubits:
            match (self.map.get(qubit, "I"), rhs.map.get(qubit, "I")):
                case ("I", "I"):
                    pass
                case ("I", p) | (p, "I"):
                    result_map[qubit] = p
                case ("X", "Y"):
                    result_map[qubit] = "Z"
                    coef *= 1j
                case ("X", "X"):
                    result_map[qubit] = "I"
                case ("X", "Z"):
                    result_map[qubit] = "Y"
                    coef *= -1j
                case ("Y", "X"):
                    result_map[qubit] = "Z"
                    coef *= -1j
                case ("Y", "Y"):
                    result_map[qubit] = "I"
                case ("Y", "Z"):
                    result_map[qubit] = "X"
                    coef *= 1j
                case ("Z", "X"):
                    result_map[qubit] = "Y"
                    coef *= 1j
                case ("Z", "Y"):
                    result_map[qubit] = "X"
                    coef *= -1j
                case ("Z", "Z"):
                    result_map[qubit] = "I"
                case _:
                    raise ValueError("Unsupported Pauli multiplication")

        return Pauli(
            None,
            None,
            _process=self.ket_process,
            _map=result_map,
            _coef=coef,
        )

    __rmul__ = __mul__

    def __len__(self) -> int:
        return len(self.map)

    def __add__(self, other: Number | Pauli | Hamiltonian) -> Hamiltonian:
        if isinstance(other, Number):
            other = Pauli(None, None, _process=self.ket_process, _map={}, _coef=other)

        _check_same_process(self, other)

        if isinstance(other, Hamiltonian):
            result = Hamiltonian([self], process=self.ket_process) + other
        else:
            result = Hamiltonian([self, other], process=self.ket_process)

        result._filter()
        return result

    __radd__ = __add__

    def __sub__(self, other: Number | Pauli | Hamiltonian) -> Hamiltonian:
        return self + -other

    def __rsub__(self, other: Number | Pauli | Hamiltonian) -> Hamiltonian:
        return -self + other

    def __truediv__(self, other: float) -> Pauli:
        return self.__mul__(1.0 / other)

    @property
    def is_bound(self) -> bool:
        """``True`` if the operator acts on qubits from a process."""
        return self.ket_process is not None

    @property
    def num_qubits(self) -> int:
        """Number of qubits needed to bind the operator.

        For an unbound operator this is the highest placeholder index plus
        one. For a bound operator it is the number of qubits it acts on.
        """
        if self.is_bound:
            return len(self.map)
        return max(self.map, default=-1) + 1

    def bind(self, qubits: Quant) -> Pauli:
        """Bind an unbound operator to qubits from a process.

        Placeholder qubit ``i`` becomes ``qubits[i]``. Calling the operator,
        ``pauli(qubits)``, is a shorthand for this method.

        Args:
            qubits: Qubits allocated from a :class:`~ket.base.Process`. Must
                have at least :attr:`num_qubits` qubits.

        Returns:
            A new operator bound to the process of ``qubits``.

        Raises:
            ValueError: If the operator is already bound or ``qubits`` is too short.
        """
        if self.is_bound:
            raise ValueError("Operator is already bound to a process")
        qubits = _as_quant(qubits)
        if len(qubits) < self.num_qubits:
            raise ValueError(
                f"Operator needs {self.num_qubits} qubits, but got {len(qubits)}"
            )
        return Pauli(
            None,
            None,
            _process=qubits.ket_process,
            _map={qubits.qubits[i]: p for i, p in self.map.items()},
            _coef=self.coef,
        )

    __call__ = bind

    @staticmethod
    def x(qubits: Quant | ObsQubits) -> Pauli:
        """Pauli X operator.

        Args:
            qubits: Qubits to apply the Pauli operator to.
        """
        return Pauli("X", qubits)

    @staticmethod
    def y(qubits: Quant | ObsQubits) -> Pauli:
        """Pauli Y operator.

        Args:
            qubits: Qubits to apply the Pauli operator to.
        """
        return Pauli("Y", qubits)

    @staticmethod
    def z(qubits: Quant | ObsQubits) -> Pauli:
        """Pauli Z operator.

        Args:
            qubits: Qubits to apply the Pauli operator to.
        """
        return Pauli("Z", qubits)

    @staticmethod
    def i(qubits: Quant | ObsQubits) -> Pauli:
        """Pauli I operator.

        Args:
            qubits: Qubits to apply the Pauli I operator to.
        """
        return Pauli("I", qubits)

    def _str_no_coef(self) -> str:
        return "".join(f"{pauli}{qubit}" for qubit, pauli in sorted(self.map.items()))

    def _latex_no_coef(self) -> str:
        return "".join(f"{pauli}_{qubit}" for qubit, pauli in sorted(self.map.items()))

    def __str__(self) -> str:
        return f"{self.coef}" + ("*" + self._str_no_coef() if len(self.map) > 0 else "")

    def __repr__(self) -> str:
        return f"<Ket 'Pauli' {str(self)}, {_pid(self.ket_process)}>"

    def _repr_latex_no_math_(self) -> str:  # pylint: disable=too-many-branches
        coef = self.coef

        def format_part(x):
            frac = Fraction(x).limit_denominator(100)
            if abs(float(frac) - x) < 1e-6:
                if frac.denominator == 1:
                    return str(frac.numerator)
                return f"\\frac{{{frac.numerator}}}{{{frac.denominator}}}"
            return str(x)

        real = coef.real
        imag = coef.imag

        parts = []

        if abs(real) > 1e-12:
            parts.append(format_part(real))

        if abs(imag) > 1e-12:
            imag_str = format_part(abs(imag))

            if imag_str == "1":
                imag_str = "i"
            else:
                imag_str = f"{imag_str}i"

            if imag < 0:
                parts.append(f"- {imag_str}")
            else:
                if parts:
                    parts.append(f"+ {imag_str}")
                else:
                    parts.append(imag_str)

        if len(parts) > 1:
            coef_str = f"({ ' '.join(parts) })"
        elif parts:
            coef_str = parts[0]
        else:
            coef_str = "0"

        pauli_str = self._latex_no_coef()

        if pauli_str:
            if coef_str == "1":
                return pauli_str
            if coef_str == "-1":
                return f"-{pauli_str}"

        return f"{coef_str}{pauli_str}"

    def _repr_latex_(self) -> str:
        return f"${self._repr_latex_no_math_()}$"


class Hamiltonian(HasProcess):
    """A sum of weighted Pauli operator products representing a quantum observable.

    A :class:`~ket.expv.Hamiltonian` is a linear combination of
    :class:`~ket.expv.Pauli` terms:

    .. math::

        H = \\sum_k c_k \\, P_k

    where each :math:`P_k` is a tensor product of single-qubit Pauli
    operators (:math:`X`, :math:`Y`, :math:`Z`, or :math:`I`) and
    :math:`c_k` is a real or complex scalar coefficient.

    .. note::
        Do not instantiate this class directly. Create Hamiltonians by
        adding :class:`~ket.expv.Pauli` objects together, or by using the
        :func:`~ket.gates.obs` context manager:

        .. code-block:: python

            from ket import *
            p = Process()
            q = p.alloc(3)
            with obs():
                # TFIM Hamiltonian: -J \u03a3 Z_i Z_{i+1} - h \u03a3 X_i
                J, h = 1.0, 0.5
                zz = sum(-J * Z(q[i]) * Z(q[i+1]) for i in range(2))
                xx = sum(-h * X(q[i]) for i in range(3))
                H = zz + xx

    To build a Hamiltonian before allocating qubits, use ``with obs(n) as q:``.
    The result is *unbound*; bind it with :meth:`bind`, the call shorthand
    ``H(qubits)``, or by passing the qubits to
    :func:`~ket.operations.exp_value` or :func:`~ket.gates.evolve`:

    .. code-block:: python

        with obs(3) as q:
            H = sum(-Z(q[i]) * Z(q[i+1]) for i in range(2))

        p = Process()
        r = p.alloc(3)
        exp_value(H, r)      # same as exp_value(H(r))

    Supported arithmetic operators:

    - **Addition** (``+``): Combine two Hamiltonians or add a scalar constant.
    - **Subtraction** (``-``): Subtract a Hamiltonian or scalar.
    - **Multiplication** (``*``): Scale by a scalar, or form the operator
      product with another Hamiltonian/Pauli.
    - **Matrix multiplication** (``@``): Full Pauli-algebra operator product.
    - **Power** (``**``): Repeated operator product (:math:`H^n`).
    - **Division** (``/``): Scale by the reciprocal of a scalar.
    - **Negation** (``-H``): Negate all coefficients.

    """

    def __init__(self, terms: list[Pauli], process: Process):
        super().__init__(ket_process=process)
        self.terms = terms
        assert all(isinstance(p, Pauli) for p in terms)

    def __add__(self, other: Hamiltonian | Pauli | Number) -> Hamiltonian:
        if isinstance(other, Number):
            other = Pauli(None, None, _process=self.ket_process, _map={}, _coef=other)
        if isinstance(other, Pauli):
            other = Hamiltonian([other], self.ket_process)

        _check_same_process(self, other)

        result = Hamiltonian(self.terms + other.terms, self.ket_process)
        result._filter()

        return result

    def __sub__(self, other: Hamiltonian | Pauli | Number) -> Hamiltonian:
        return self + -other

    __radd__ = __add__

    def _filter(self):
        # Merge like terms into copies, so the operands' terms are never modified.
        new_terms = {}
        for term in self.terms:
            str_term = term._str_no_coef()
            if all(c not in str_term for c in "XYZ"):
                str_term = ""
            if str_term not in new_terms:
                new_terms[str_term] = Pauli(
                    None,
                    None,
                    _process=term.ket_process,
                    _map=dict(term.map),
                    _coef=term.coef,
                )
            else:
                new_terms[str_term].coef += term.coef

        self.terms = [term for term in new_terms.values() if abs(term.coef) > 1e-10]

    def __matmul__(self, other: Hamiltonian | Pauli) -> Hamiltonian:
        if isinstance(other, Pauli):
            other = Hamiltonian([other], self.ket_process)

        _check_same_process(self, other)

        result = Hamiltonian(
            [a @ b for a, b in product(self.terms, other.terms)],
            self.ket_process,
        )

        result._filter()

        return result

    def __rsub__(self, other: Hamiltonian | Pauli | Number) -> Hamiltonian:
        return -self + other

    def __mul__(
        self, other: float | complex | Hamiltonian | Pauli | Parameter
    ) -> Hamiltonian:
        if isinstance(other, (Number, Pauli, Parameter)):
            return Hamiltonian(
                [p * other for p in self.terms], process=self.ket_process
            )

        if isinstance(other, Hamiltonian):
            _check_same_process(self, other)
            return Hamiltonian(
                [p * q for p, q in product(self.terms, other.terms)],
                process=self.ket_process,
            )

        raise TypeError(
            f"Unsupported type for multiplication: {type(other)}. "
            "Expected float, Hamiltonian, or Pauli."
        )

    __rmul__ = __mul__

    def __pow__(self, exp: int) -> Hamiltonian:
        exp = int(exp)
        if exp == 0:
            return Hamiltonian(
                [Pauli(None, None, _process=self.ket_process, _map={}, _coef=1.0)],
                self.ket_process,
            )

        result = self
        for _ in range(exp - 1):
            result = result @ self
        return result

    def __truediv__(self, other: float) -> Hamiltonian:
        return self.__mul__(1.0 / other)

    def __neg__(self) -> Hamiltonian:
        return -1.0 * self

    def __len__(self) -> int:
        return len(self.terms)

    @property
    def is_bound(self) -> bool:
        """``True`` if the Hamiltonian acts on qubits from a process."""
        return self.ket_process is not None

    @property
    def num_qubits(self) -> int:
        """Number of qubits needed to bind the Hamiltonian.

        For an unbound Hamiltonian this is the highest placeholder index plus
        one. For a bound Hamiltonian it is the number of qubits it acts on.
        """
        if self.is_bound:
            return len({q for term in self.terms for q in term.map})
        return max((term.num_qubits for term in self.terms), default=0)

    def bind(self, qubits: Quant) -> Hamiltonian:
        """Bind an unbound Hamiltonian to qubits from a process.

        Placeholder qubit ``i`` becomes ``qubits[i]``. Calling the Hamiltonian,
        ``h(qubits)``, is a shorthand for this method, so an unbound
        Hamiltonian can be passed anywhere a ``Callable[[Quant], Hamiltonian]``
        is expected.

        Args:
            qubits: Qubits allocated from a :class:`~ket.base.Process`. Must
                have at least :attr:`num_qubits` qubits.

        Returns:
            A new Hamiltonian bound to the process of ``qubits``.

        Raises:
            ValueError: If the Hamiltonian is already bound or ``qubits`` is too short.
        """
        if self.is_bound:
            raise ValueError("Hamiltonian is already bound to a process")
        qubits = _as_quant(qubits)
        if len(qubits) < self.num_qubits:
            raise ValueError(
                f"Hamiltonian needs {self.num_qubits} qubits, but got {len(qubits)}"
            )
        return Hamiltonian(
            [term.bind(qubits) for term in self.terms], qubits.ket_process
        )

    __call__ = bind

    def __repr__(self) -> str:
        return (
            f"<Ket 'Hamiltonian' {' + '.join(str(p) for p in self.terms)}, "
            f"{_pid(self.ket_process)}>"
        )

    def _repr_latex_(self) -> str:
        return "$" + " + ".join(p._repr_latex_no_math_() for p in self.terms) + "$"


def commutator(a: Hamiltonian, b: Hamiltonian) -> Hamiltonian:
    r"""Calculate the commutator of two Hamiltonians.

    Computes :math:`[A, B] = AB - BA` using the :class:`~ket.expv.Hamiltonian`
    matrix-multiplication operator (``@``).

    The commutator is zero if and only if the two operators share a common
    eigenbasis, i.e., they can be measured simultaneously. A non-zero
    commutator implies Heisenberg uncertainty between the corresponding
    observables.

    Example:

        .. code-block:: python

            from ket import *
            p = Process()
            q = p.alloc()
            with obs():
                x = X(q)
                z = Z(q)
            xz_comm = commutator(x, z)    # [X, Z] = -2iY
            print(xz_comm)                # displays the Pauli Y term

    Args:
        a: Left-hand operator.
        b: Right-hand operator.

    Returns:
        The commutator :math:`[A, B]`.
    """
    return a @ b - b @ a


class ExpValue(HasProcess):
    """A deferred handle for the expectation value of a Hamiltonian.

    Registering an expectation value does not immediately compute it,
    in **live** execution mode, the value is computed right away; in **batch**
    execution mode it is deferred and becomes available only after
    :meth:`~ket.expv.ExpValue.get` (or a measurement) triggers process
    execution.

    .. note::
        Do not instantiate this class directly. Use
        :func:`~ket.operations.exp_value` instead.

    Example:

        .. code-block:: python

            from ket import *
            p = Process()
            q = p.alloc(2)
            CNOT(H(q[0]), q[1])    # Bell state |Φ+⟩
            with obs():
                observable = X(q[0]) * X(q[1]) - Y(q[0]) * Y(q[1])
            ev = exp_value(observable)
            print(ev.get())        # 2.0 for the Bell state

    """

    pauli_map = {"X": "PauliX", "Y": "PauliY", "Z": "PauliZ"}

    def __init__(self, hamiltonian: Hamiltonian | Pauli, qubits: Quant | None = None):
        if qubits is not None:
            hamiltonian = hamiltonian.bind(qubits)
        elif not hamiltonian.is_bound:
            raise ValueError(
                "Cannot compute the expectation value of an unbound Hamiltonian; "
                "pass the qubits to exp_value or bind it first"
            )
        super().__init__(ket_process=hamiltonian.ket_process)
        if isinstance(hamiltonian, Pauli):
            hamiltonian = Hamiltonian([hamiltonian], process=hamiltonian.ket_process)

        h_json = {
            "pauli_strings": [],
            "coefficients": [],
        }

        products_count = 0
        self._i_coef = 0.0

        for pauli_product in hamiltonian.terms:
            if not pauli_product.map:
                self._i_coef += pauli_product.coef.real
                continue

            if (
                isinstance(pauli_product.coef, complex)
                and abs(pauli_product.coef.imag) > 1e-10
            ):
                raise ValueError(
                    "Complex coefficients are not supported in Hamiltonian"
                    " expected value calculation."
                )

            pauli_qubits = [
                (self.pauli_map[p], q) for q, p in pauli_product.map.items() if p != "I"
            ]

            if not pauli_qubits:
                self._i_coef += pauli_product.coef.real
                continue

            term = [{"pauli": p, "qubit": q} for p, q in pauli_qubits]

            h_json["pauli_strings"].append(term)
            h_json["coefficients"].append(pauli_product.coef.real)

            products_count += 1

        if products_count > 0:
            self.ket_process._check_not_in_block("exp_value")
            self.index = self.ket_process._get_exp_value_index()
            result, some_result = self.ket_process.exp_value(
                json.dumps(h_json).encode("utf-8")
            )
            if some_result:
                self._value = result.value + self._i_coef
            else:
                self._value = None
        else:
            self._value = self._i_coef

    def _check(self, execute=False):
        if self._value is None:
            result = self.ket_process._get_exp_value(self.index, execute)
            if result is not None:
                self._value = result + self._i_coef

    @property
    def value(self) -> float | None:
        """The expectation value if available, otherwise ``None``.

        In **live** execution mode this is populated immediately after
        :class:`~ket.expv.ExpValue` is created. In **batch** mode it is
        ``None`` until the process executes.

        Returns:
            The computed expectation value, or ``None`` if the
            process has not yet executed.
        """
        self._check()
        return self._value

    def get(self) -> float:
        """Retrieve the expectation value, executing the process if necessary.

        Blocks until the expectation value is available. In **batch** execution
        mode this call triggers process execution.

        Returns:
            The computed expectation value.
        """
        self._check(execute=True)
        return self.value

    def __repr__(self) -> str:
        return f"<Ket 'ExpValue' value={self.value}, pid={hex(id(self.ket_process))}>"
