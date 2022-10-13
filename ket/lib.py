from __future__ import annotations
#  Copyright 2020, 2021 Evandro Chagas Ribeiro da Rosa <evandro.crr@posgrad.ufsc.br>
#  Copyright 2020, 2021 Rafael de Santiago <r.santiago@ufsc.br>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from math import pi, sqrt, asin
from typing import Callable

from .base import quant, dump, future
from .gates import H, phase, swap, cnot, X, Y, Z, S, RY
from .standard import control, ctrl
from .process import run, exec_quantum
from .preprocessor.statements import _ket_if, _ket_next


def qft(qubits: quant, invert: bool = True) -> quant:
    """Quantum Fourier Transformation

    Apply a QFT_ on the qubits of q.

    .. _QFT: https://en.wikipedia.org/wiki/Quantum_Fourier_transform

    Args:
        q: input qubits
        invert: if ``True``, invert qubits with swap gates

    return:
        Qubits in the reserved order
    """

    if len(qubits) == 1:
        H(qubits)
    else:
        head, *tail = qubits
        H(head)
        for i, ctrl_qubit in enumerate(tail):
            with control(ctrl_qubit):
                phase(2 * pi / 2**(i + 2), head)
        qft(tail, invert=False)

    if invert:
        for i in range(len(qubits) // 2):
            swap(qubits[i], qubits[len(qubits) - i - 1])
        return qubits
    return reversed(qubits)


def dump_matrix(gate: Callable | list[Callable], size: int = 1) -> list[list[complex]]:
    """Get the matrix of a quantum operation

    Args:
        u: Quantum operation.
        size: Number of qubits.
    """

    mat = [[0 for _ in range(2**size)] for _ in range(2**size)]
    with run():
        row = quant(size)
        column = quant(size)
        H(column)
        cnot(column, row)
        if hasattr(gate, '__iter__'):
            for func in gate:
                func(row)
        else:
            gate(row)
        state = dump(column + row)
        exec_quantum()

    for state, amp in zip(state.states, state.amplitudes):
        column = state >> size
        row = state & ((1 << size) - 1)
        mat[row][column] = amp * sqrt(2**size)

    return mat


def bell(x: int = 0, y: int = 0, qubits: quant | None = None) -> quant:  # pylint: disable=invalid-name
    r"""Bell state preparation

    Return two entangle qubits in the Bell state:

    .. math::

        \left|\beta_{x,y}\right> = \frac{\left|0,y\right> + (-1)^x\left|1,¬y\right>}{\sqrt{2}}

    Args:
        x: :math:`x`.
        y: :math:`y`.
        qubits: if provided, prepare the state in ``qubits``.
    """

    if qubits is not None and (len(qubits) != 2 or not isinstance(qubits, quant)):
        raise AttributeError(
            "if param 'qubits' is provided, it must be a quant of length 2")

    if qubits is None:
        qubits = quant(2)

    if isinstance(x, future):
        end = _ket_if(x)  # pylint: disable=undefined-variable
        X(qubits[0])
        _ket_next(end)  # pylint: disable=undefined-variable
    elif x:
        X(qubits[0])

    if isinstance(y, future):
        end = _ket_if(y)  # pylint: disable=undefined-variable
        X(qubits[1])
        _ket_next(end)  # pylint: disable=undefined-variable
    elif y:
        X(qubits[1])

    H(qubits[0])
    ctrl(qubits[0], X, qubits[1])
    return qubits


def ghz(qubits: quant | int) -> quant:
    r"""GHZ state preparation

    Return qubits on the state:

    .. math::

        \left|\text{GHZ}\right> = \frac{\left|0\dots0\right> + \left|1\dots1\right>}{\sqrt{2}}

    args:
        qubits: Qubits to prepare the state, if it is an ``int`` allocate the number of qubits.
    """

    if not isinstance(qubits, quant):
        qubits = quant(qubits)

    ctrl(H(qubits[0]), X, qubits[1:])

    return qubits


def w(qubits: quant | int) -> quant:  # pylint: disable=invalid-name
    r"""W state preparation

    Return qubits on the state:

    .. math::

        \left|\text{W}\right> = \frac{1}{\sqrt{n}} \sum_{k=0}^{n} \left|2^k\right>

    args:
        qubits: Qubits to prepare the state, if it is an ``int`` allocate the number of qubits.
    """

    if not isinstance(qubits, quant):
        size = qubits
        qubits = quant(qubits)
    size = len(qubits)

    X(qubits[0])
    for i in range(size - 1):
        with control(qubits[i]):
            n = size - i  # pylint: disable=invalid-name
            RY(2 * asin(sqrt((n - 1) / n)), qubits[i + 1])
        cnot(qubits[i + 1], qubits[i])

    return qubits


def pauli(basis: Callable, qubit: quant, state: int = +1) -> quant:  # pylint: disable=redefined-outer-name
    """Prepare in the +1 or -1 eigenstate of a given Pauli operator

    Args:
        basis: Pauli operator :func:`~ket.gates.X`, :func:`~ket.gates.Y`, or :func:`~ket.gates.Z`.
        q: Input qubits.
        state: Eigenstate.
    """

    if isinstance(state, future):
        end = _ket_if(state == -1)  # pylint: disable=undefined-variable
        X(qubit)
        _ket_next(end)  # pylint: disable=undefined-variable
    elif state == -1:
        X(qubit)
    elif state == 1:
        pass
    else:
        raise ValueError("Param 'state' must be +1 or -1.")

    if basis == X:
        return H(qubit)
    if basis == Y:
        return S(H(qubit))
    if basis == Z:
        return qubit
    raise ValueError("Param 'basis' must be X, Y, or Z.")
