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

from .. import *
from .. import code_ket
from math import pi, sqrt
from typing import Callable, Iterable

def qft(q : quant, invert : bool = True) -> quant:
    """Quantum Fourier Transformation
    
    Apply a QFT_ on the qubits of q.

    Note:
        Implace operation.

    .. _QFT: https://en.wikipedia.org/wiki/Quantum_Fourier_transform

    :param q: input qubits
    :param invert: if ``True``, invert qubits with swap gates
    :return: qubits in the reserved order 
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

def dump_matrix(u : Callable | Iterable[Callable], size : int = 1) -> list[list[complex]]:
    """Get the matrix of a quantum operation.
    
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
            ret = []
            for ui in u:
               ret.append(ui(row)) 
        else:
            ret = u(row)
        d = dump(column+row)
        exec_quantum()
    
    for state in d.states:
        column = state >> size
        row = state & ((1 << size)-1)
        mat[row][column] = d.amplitude(state)*sqrt(2**size)

    if ret != None:
        return mat, ret
    else:
        return mat
     