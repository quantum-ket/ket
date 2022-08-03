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
from . import *
from .preprocessor.statements import *


def qft(q: quant, invert: bool = True) -> quant:
    """Quantum Fourier Transformation

    Apply a QFT_ on the qubits of q.

    .. _QFT: https://en.wikipedia.org/wiki/Quantum_Fourier_transform

    Args:
        q: input qubits
        invert: if ``True``, invert qubits with swap gates

    return:
        Qubits in the reserved order 
    """

    if len(q) == 1:
        H(q)
    else:
        head, *tail = q
        H(head)
        for i in range(len(tail)):
            with control(tail[i]):
                phase(2*pi/2**(i+2), head)
        qft(tail, invert=False)

    if invert:
        for i in range(len(q)//2):
            swap(q[i], q[len(q)-i-1])
        return q
    else:
        return reversed(q)


def dump_matrix(u: Callable | list[Callable], size: int = 1) -> list[list[complex]]:
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
        if hasattr(u, '__iter__'):
            for ui in u:
                ui(row)
        else:
            u(row)
        d = dump(column+row)
        exec_quantum()

    for state, amp in zip(d.states, d.amplitudes):
        column = state >> size
        row = state & ((1 << size)-1)
        mat[row][column] = amp*sqrt(2**size)

    return mat


def bell(x: int = 0, y: int = 0, qubits: quant | None = None) -> quant:
    r"""Bell state preparation

    Return two entangle qubits in the Bell state:

    .. math::

        \left|\beta_{x,y}\right> = \frac{\left|0,y\right> + (-1)^x\left|1,¬y\right>}{\sqrt{2}}

    Args: 
        x: :math:`x`.
        y: :math:`y`.
        qubits: if provided, prepare the state in ``qubits``.
    """

    if qubits is not None and (len(qubits) != 2 or type(qubits) != quant):
        raise AttributeError(
            "if param 'qubits' is provided, it must be a quant of length 2")

    if qubits is None:
        qubits = quant(2)

    if isinstance(x, future):
        end = _ket_if(x)
        X(qubits[0])
        _ket_next(end)
    elif x:
        X(qubits[0])

    if isinstance(y, future):
        end = _ket_if(y)
        X(qubits[1])
        _ket_next(end)
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


def w(qubits: quant | int) -> quant:
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
    for i in range(size-1):
        with control(qubits[i]):
            n = size - i
            RY(2*asin(sqrt((n-1)/n)), qubits[i+1])
        cnot(qubits[i+1], qubits[i])

    return qubits


def pauli(basis: Callable, q: quant, state: int = +1) -> quant:
    """Prepare in the +1 or -1 eigenstate of a given Pauli operator

    Args:
        basis: Pauli operator :func:`~ket.gates.X`, :func:`~ket.gates.Y`, or :func:`~ket.gates.Z`.
        q: Input qubits.
        state: Eigenstate.
    """

    if isinstance(state, future):
        end = _ket_if(state == -1)
        X(q)
        _ket_next(end)
    elif state == -1:
        X(q)
    elif state == 1:
        pass
    else:
        raise ValueError("Param 'state' must be +1 or -1.")

    if basis == X:
        return H(q)
    elif basis == Y:
        return S(H(q))
    elif basis == Z:
        return q
    else:
        raise ValueError("Param 'basis' must be X, Y, or Z.")
