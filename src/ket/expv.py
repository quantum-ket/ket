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
        print(ev.get())          # -3.0 for the singlet Bell state
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=duplicate-code,protected-access
import json
from numbers import Number
from itertools import product
from functools import reduce
from typing import Literal
from fractions import Fraction

from .base import Parameter, Process, Quant
from .clib.libket import HasProcess

__all__ = [
    "Pauli",
    "Hamiltonian",
    "commutator",
    "ExpValue",
]


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
        qubits: Qubits to apply the Pauli operator to.
        process: *For internal usage*. Quantum process, default is the process of the given qubits.
        pauli_list: *For internal usage*. List of Pauli operators.
        qubits_list: *For internal usage*. List of Qubit.
        coef: *For internal usage*. Coefficient for the Pauli operator, default is 1.0.

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pauli: Literal["X", "Y", "Z", "I"] | None,
        qubits: Quant | None,
        *,
        _process: Process | None = None,
        _map: dict[int, str] | None = None,
        _coef: float | complex | None = None,
    ):
        if qubits is not None and not isinstance(qubits, Quant):
            qubits = reduce(Quant.__add__, qubits)

        if pauli is not None and qubits is not None:
            super().__init__(ket_process=qubits.ket_process)
            self.map = {q: pauli for q in qubits.qubits}
            self.coef = 1.0
        else:
            assert _process is not None and _map is not None and _coef is not None, (
                "If pauli and qubits are not provided, "
                "process, map, and coef must be provided."
            )
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

        if self.ket_process is not other.ket_process:
            raise ValueError("different Ket processes")

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

        if self.ket_process is not rhs.ket_process:
            raise ValueError("different Ket processes")

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

        if self.ket_process is not other.ket_process:
            raise ValueError("different Ket processes")

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

    @staticmethod
    def x(qubits: Quant) -> Pauli:
        """Pauli X operator.

        Args:
            qubits: Qubits to apply the Pauli operator to.
        """
        return Pauli("X", qubits)

    @staticmethod
    def y(qubits: Quant) -> Pauli:
        """Pauli Y operator.

        Args:
            qubits: Qubits to apply the Pauli operator to.
        """
        return Pauli("Y", qubits)

    @staticmethod
    def z(qubits: Quant) -> Pauli:
        """Pauli Z operator.

        Args:
            qubits: Qubits to apply the Pauli operator to.
        """
        return Pauli("Z", qubits)

    @staticmethod
    def i(qubits: Quant) -> Pauli:
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
        return f"<Ket 'Pauli' {str(self)}, pid={hex(id(self.ket_process))}>"

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

        if self.ket_process is not other.ket_process:
            raise ValueError("different Ket processes")

        result = Hamiltonian(self.terms + other.terms, self.ket_process)
        result._filter()

        return result

    def __sub__(self, other: Hamiltonian | Pauli | Number) -> Hamiltonian:
        return self + -other

    __radd__ = __add__

    def _filter(self):
        new_terms = {}
        for term in self.terms:
            str_term = term._str_no_coef()
            if all(c not in str_term for c in "XYZ"):
                str_term = ""
            if str_term not in new_terms:
                new_terms[str_term] = term
            else:
                new_terms[str_term].coef += term.coef

        self.terms = [term for term in new_terms.values() if abs(term.coef) > 1e-10]

    def __matmul__(self, other: Hamiltonian | Pauli) -> Hamiltonian:
        if isinstance(other, Pauli):
            other = Hamiltonian([other], self.ket_process)

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

    def __repr__(self) -> str:
        return (
            f"<Ket 'Hamiltonian' {' + '.join(str(p) for p in self.terms)}, "
            f"pid={hex(id(self.ket_process))}>"
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

    def __init__(self, hamiltonian: Hamiltonian | Pauli):
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
            self._value = (
                self.ket_process._get_exp_value(self.index, execute) + self._i_coef
            )

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
