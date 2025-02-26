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
from functools import reduce
from operator import add
from typing import Literal

from .base import Process, Quant

from .clib.libket import API

__all__ = [
    "Pauli",
    "Hamiltonian",
    "ExpValue",
]


class Pauli:
    """Pauli operator for Hamiltonian creation.

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
        _pauli_list: list[str] | None = None,
        _qubits_list: list[Quant] | None = None,
        _coef: float | None = None,
    ):
        if not isinstance(qubits, Quant) and _qubits_list is None:
            qubits = reduce(add, qubits)

        self.process = _process if _process is not None else qubits.process
        self.pauli_list = _pauli_list if _pauli_list is not None else [pauli]
        self.qubits_list = _qubits_list if _qubits_list is not None else [qubits]
        self.coef = 1.0 if _coef is None else _coef

    def _flat(self) -> tuple[list[str], list[Quant]]:
        pauli_list = []
        qubits_list = []
        for pauli, qubits in zip(self.pauli_list, self.qubits_list):
            pauli_list += [pauli] * len(qubits.qubits)
            qubits_list += qubits.qubits
        return pauli_list, qubits_list

    def __neg__(self) -> Pauli:
        return -1.0 * self

    def __mul__(self, other: float | Pauli) -> Pauli:
        if isinstance(other, (float, int)):
            return Pauli(
                None,
                None,
                _process=self.process,
                _pauli_list=self.pauli_list,
                _qubits_list=self.qubits_list,
                _coef=self.coef * other,
            )
        if isinstance(other, Hamiltonian):
            return other.__mul__(self)

        if self.process is not other.process:
            raise ValueError("different Ket processes")

        return Pauli(
            None,
            None,
            _process=self.process,
            _pauli_list=self.pauli_list + other.pauli_list,
            _qubits_list=self.qubits_list + other.qubits_list,
            _coef=self.coef * other.coef,
        )

    def __div__(self, other: float) -> Pauli:
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

    def __rmul__(self, other: float) -> Pauli:
        return Pauli(
            None,
            None,
            _process=self.process,
            _pauli_list=self.pauli_list,
            _qubits_list=self.qubits_list,
            _coef=self.coef * other,
        )

    def __add__(self, other) -> Hamiltonian:
        if self.process is not other.process:
            raise ValueError("different Ket processes")

        return Hamiltonian([self, other], process=self.process)

    def __sub__(self, other) -> Hamiltonian:
        return self + (-1 * other)

    def __radd__(self, other: int | float) -> Pauli:
        if other != 0:
            raise ValueError("cannot add Pauli with float or int")

        return self

    def __str__(self) -> str:
        return f"{self.coef}*" + "".join(
            "".join(f"{pauli}{qubit}" for qubit in qubits.qubits)
            for pauli, qubits in zip(self.pauli_list, self.qubits_list)
        )

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

    def __add__(self, other: Hamiltonian | Pauli) -> Hamiltonian:
        if isinstance(other, Pauli):
            other = Hamiltonian([other], self.process)
            return self + other

        if self.process is not other.process:
            raise ValueError("different Ket processes")

        return Hamiltonian(self.pauli_products + other.pauli_products, self.process)

    def __sub__(self, other: Hamiltonian | Pauli) -> Hamiltonian:
        return self + (-1 * other)

    def __radd__(self, other: int | float) -> Hamiltonian:
        if other != 0:
            raise ValueError("cannot add Hamiltonian with float or int")

        return self

    def __mul__(self, other: float) -> Hamiltonian:
        return Hamiltonian(
            [p * other for p in self.pauli_products], process=self.process
        )

    __rmul__ = __mul__

    def __truediv__(self, other: float) -> Hamiltonian:
        return self.__mul__(1.0 / other)

    def __neg__(self) -> Hamiltonian:
        return -1.0 * self

    def __repr__(self) -> str:
        return (
            f"<Ket 'Hamiltonian' {' + '.join(str(p) for p in self.pauli_products)}, "
            f"pid={hex(id(self.process))}>"
        )


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
            pauli, qubits = pauli_product._flat()
            pauli_qubits = list(
                zip(
                    *[(self.pauli_map[p], q) for p, q in zip(pauli, qubits) if p != "I"]
                )
            )
            if not pauli_qubits:
                self._i_coef += pauli_product.coef
                continue
            pauli, qubits = pauli_qubits
            API["ket_hamiltonian_add"](
                hamiltonian_ptr,
                (c_int32 * len(pauli))(*pauli),
                len(pauli),
                (c_size_t * len(qubits))(*qubits),
                len(qubits),
                pauli_product.coef,
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
