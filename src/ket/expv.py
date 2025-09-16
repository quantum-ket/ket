"""Expected value calculation utilities.

This module provides base classes creating a Hamiltonian fro expected value calculation.
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=duplicate-code

from ctypes import c_int32, c_size_t
from numbers import Number
from itertools import product
from functools import reduce
from operator import add
from typing import Literal

from .base import Process, Quant

from .clib.libket import API

__all__ = [
    "Pauli",
    "Hamiltonian",
    "commutator",
    "ExpValue",
]


class Pauli:
    """Pauli operator for Hamiltonian creation.

    .. tip::
        The preferred way to create a Hamiltonian is by using the
        :func:`~ket.gates.ham` context manager, which allows you to define a Hamiltonian
        using a more intuitive syntax. However, you can also create Hamiltonians directly by
        instantiating this class.

    This class represents a Pauli operator for Hamiltonian creation. The primary usage of this class
    is to prepare a Hamiltonian by adding and multiplying Pauli operators and scalars for
    calculating the expected value of a quantum state.

    Allowed operations:
    - Multiply by a scalar or another :class:`~ket.expv.Pauli` operator.
    - Add another :class:`~ket.expv.Pauli` or :class:`~ket.expv.Hamiltonian` operator.

    Example:

        .. code-block::

            from ket import *

            p = Process()
            qubits = p.alloc(3)
            H(qubits[1])
            S(H(qubits[2]))

            # Example 1: Single Pauli term
            pauli_term = Pauli("X", qubits[0])
            print(repr(pauli_term))
            # <Ket 'Pauli' 1.0*X0, pid=0x...>

            # Example 2: Sum of Pauli terms
            sum_pauli = Pauli("Y", qubits[1]) + Pauli("Z", qubits[2])
            print(repr(sum_pauli))
            # <Ket 'Hamiltonian' 1.0*Y1 + 1.0*Z2, pid=0x...>

            # Example 3: Pauli multiplication
            multiplied_pauli = 2.0 * Pauli("X", qubits[0]) * Pauli("Y", qubits[1])
            print(repr(multiplied_pauli))
            # <Ket 'Pauli' 2.0*X0Y1, pid=0x...>

            # Example 4: Calculating expected value
            h = Pauli("Z", qubits[0]) + Pauli("X", qubits[1]) + Pauli("Y", qubits[2])
            print(repr(h))
            # <Ket 'Hamiltonian' 1.0*Z0 + 1.0*X1 + 1.0*Y2, pid=0x...>
            result = exp_value(h)
            print(result.value)  # 3.0000000000000013


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
        pauli: Literal["X", "Y", "Z", "I"],
        qubits: Quant,
        *,
        _process: Process | None = None,
        _map: dict[int, str] | None = None,
        _coef: float | None = None,
    ):
        if qubits is not None and not isinstance(qubits, Quant):
            qubits = reduce(add, qubits)

        if pauli is not None and qubits is not None:
            self.process = qubits.process
            self.map = {q: pauli for q in qubits.qubits}
            self.coef = 1.0
        else:
            assert _process is not None and _map is not None and _coef is not None, (
                "If pauli and qubits are not provided, "
                "process, map, and coef must be provided."
            )

            self.process = _process
            self.map = _map
            self.coef = _coef

    def __neg__(self) -> Pauli:
        return -1.0 * self

    def __mul__(self, other: float | Pauli) -> Pauli:

        if isinstance(other, Number):
            return Pauli(
                None,
                None,
                _process=self.process,
                _map=self.map,
                _coef=self.coef * other,
            )

        if isinstance(other, Hamiltonian):
            return other.__mul__(self)

        if self.process is not other.process:
            raise ValueError("different Ket processes")

        if not set(self.map.keys()).isdisjoint(other.map.keys()):
            raise ValueError("Pauli operators act on the same qubits")

        return Pauli(
            None,
            None,
            _process=self.process,
            _map={**self.map, **other.map},
            _coef=self.coef * other.coef,
        )

    def __matmul__(self, rhs: Pauli) -> Pauli:
        if isinstance(rhs, Hamiltonian):
            return Hamiltonian([self], self.process) @ rhs

        if self.process is not rhs.process:
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
            _process=self.process,
            _map=result_map,
            _coef=coef,
        )

    __rmul__ = __mul__

    def __len__(self) -> int:
        return len(self.map)

    def __add__(self, other) -> Hamiltonian:
        if isinstance(other, Number):
            other = Pauli(None, None, _process=self.process, _map={}, _coef=other)

        if self.process is not other.process:
            raise ValueError("different Ket processes")

        if isinstance(other, Hamiltonian):
            result = Hamiltonian([self], process=self.process) + other
        else:
            result = Hamiltonian([self, other], process=self.process)

        result._filter()
        return result

    __radd__ = __add__

    def __sub__(self, other) -> Hamiltonian:
        return self + -other

    def __rsub__(self, other) -> Hamiltonian:
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

    def __str__(self) -> str:
        return f"{self.coef}" + ("*" + self._str_no_coef() if len(self.map) > 0 else "")

    def __repr__(self) -> str:
        return f"<Ket 'Pauli' {str(self)}, pid={hex(id(self.process))}>"


class Hamiltonian:
    """Hamiltonian for expected value calculation.

    This class is not intended to be instantiated directly. Instead, it should be created
    by adding :class:`~ket.expv.Pauli` operators or other :class:`~ket.expv.Hamiltonian`
    objects.
    """

    def __init__(self, pauli_products: list[Pauli], process: Process):
        self.process = process
        self.pauli_products = pauli_products
        assert all(isinstance(p, Pauli) for p in pauli_products)

    def __add__(self, other: Hamiltonian | Pauli) -> Hamiltonian:
        if isinstance(other, Number):
            other = Pauli(None, None, _process=self.process, _map={}, _coef=other)
        if isinstance(other, Pauli):
            other = Hamiltonian([other], self.process)

        if self.process is not other.process:
            raise ValueError("different Ket processes")

        result = Hamiltonian(self.pauli_products + other.pauli_products, self.process)
        result._filter()

        return result

    def __sub__(self, other: Hamiltonian | Pauli) -> Hamiltonian:
        return self + -other

    __radd__ = __add__

    def _filter(self):
        new_terms = {}
        for term in self.pauli_products:
            str_term = term._str_no_coef()  # pylint: disable=protected-access
            if str_term not in new_terms:
                new_terms[str_term] = term
            else:
                new_terms[str_term].coef += term.coef

        self.pauli_products = [
            term for term in new_terms.values() if abs(term.coef) > 1e-10
        ]

    def __matmul__(self, other: Hamiltonian | Pauli) -> Hamiltonian:
        if isinstance(other, Pauli):
            other = Hamiltonian([other], self.process)

        result = Hamiltonian(
            [a @ b for a, b in product(self.pauli_products, other.pauli_products)],
            self.process,
        )

        result._filter()

        return result

    def __rsub__(self, other) -> Hamiltonian:
        return -self + other

    def __mul__(self, other: float | Hamiltonian | Pauli) -> Hamiltonian:
        if isinstance(other, (Number, Pauli)):
            return Hamiltonian(
                [p * other for p in self.pauli_products], process=self.process
            )

        if isinstance(other, Hamiltonian):
            return Hamiltonian(
                [p * q for p, q in product(self.pauli_products, other.pauli_products)],
                process=self.process,
            )

        raise TypeError(
            f"Unsupported type for multiplication: {type(other)}. "
            "Expected float, Hamiltonian, or Pauli."
        )

    __rmul__ = __mul__

    def __truediv__(self, other: float) -> Hamiltonian:
        return self.__mul__(1.0 / other)

    def __neg__(self) -> Hamiltonian:
        return -1.0 * self

    def __len__(self) -> int:
        return len(self.pauli_products)

    def __repr__(self) -> str:
        return (
            f"<Ket 'Hamiltonian' {' + '.join(str(p) for p in self.pauli_products)}, "
            f"pid={hex(id(self.process))}>"
        )


def commutator(a: Hamiltonian, b: Hamiltonian) -> Hamiltonian:
    """Calculate the commutator of two Hamiltonians."""
    return a @ b - b @ a


class ExpValue:
    """Expected value for a quantum state.

    This class holds a reference for a expected value result. The result may not be available right
    after the measurement call, especially in batch execution.

    To read the value, access the attribute :attr:`~ket.base.ExpValue.value`. If the value is not
    available, the measurement will return `None`; otherwise, it will return a float.

    You can instantiate this class by calling the :func:`~ket.operations.exp_value` function.

    Example:

        .. code-block:: python

            from ket import *

            p = Process()
            q = p.alloc(2)
            CNOT(H(q[0]), q[1])
            result = exp_value(Pauli("X", q))
            print(result.value) # 1.0000000000000004

    """

    pauli_map = {"X": 1, "Y": 2, "Z": 3}

    def __init__(self, hamiltonian: Hamiltonian | Pauli):
        if isinstance(hamiltonian, Pauli):
            hamiltonian = Hamiltonian([hamiltonian], process=hamiltonian.process)

        self.process = hamiltonian.process

        hamiltonian_ptr = API["ket_hamiltonian_new"]()
        products_count = 0
        self._i_coef = 0.0
        for pauli_product in hamiltonian.pauli_products:
            if len(pauli_product.map) == 0:
                self._i_coef += pauli_product.coef.real
                continue

            qubits, pauli = zip(*pauli_product.map.items())
            pauli_qubits = list(
                zip(
                    *[(self.pauli_map[p], q) for p, q in zip(pauli, qubits) if p != "I"]
                )
            )
            if (
                isinstance(pauli_product.coef, complex)
                and abs(pauli_product.coef.imag) > 1e-10
            ):
                raise ValueError(
                    "Complex coefficients are not supported in Hamiltonian"
                    " expected value calculation."
                )

            if not pauli_qubits:
                self._i_coef += pauli_product.coef.real
                continue
            pauli, qubits = pauli_qubits
            API["ket_hamiltonian_add"](
                hamiltonian_ptr,
                (c_int32 * len(pauli))(*pauli),
                len(pauli),
                (c_size_t * len(qubits))(*qubits),
                len(qubits),
                pauli_product.coef.real,
            )
            products_count += 1

        if products_count > 0:
            self.index = self.process.exp_value(hamiltonian_ptr).value
            self._value = None
        else:
            self._value = self._i_coef

    def _check(self):
        if self._value is None:
            available, value = self.process.get_exp_value(self.index)
            if available.value:
                self._value = value.value + self._i_coef

    @property
    def value(self) -> float | None:
        """Retrieve the expected value if available."""
        self._check()
        return self._value

    def get(self) -> float:
        """Retrieve the expected values.

        If the value is not available, the quantum process will execute to get the result.
        """

        self._check()
        if self._value is None:
            self.process.execute()
        return self.value

    def __repr__(self) -> str:
        return f"<Ket 'ExpValue' value={self.value}, pid={hex(id(self.process))}>"
