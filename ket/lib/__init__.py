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
from typing import Union, List

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
        h(q)
    else:
        head, tail = q[0], q[1:]
        h(head)
        for i in range(len(tail)):
            ctrl(tail[i], u1, 2*pi/2**(i+2), head)
        qft(tail, invert=False)

    if invert:
        for i in range(len(q)//2):
            swap(q[i], q[len(q)-i-1])
    return q.inverted()

@code_ket
def bell(x : int = 0, y : int = 0, qubits : quant = None) -> quant:
    """Bell state preparation
    
    Return two entangle qubits in the Bell state:
    
    .. math::

        \left|\beta_{x,y}\right> = \frac{\left|0,y\right> + (-1)^x\left|1,¬y\right>}{\sqrt{2}}
        
    :param x: aux 0
    :param y: aux 1
    :param qubits: if provided, prepare state in qubits; else, create a new quant
    :return: 2 qubits in the Bell state
    """

    if qubits is not None and (len(qubits) != 2 or type(qubits) != quant):
        raise AttributeError("if param 'qubits' is provided, it must be a quant of length 2")

    if qubits is None:
        qubits = quant(2)

    if x == 1:
        X(qubits[0])
    if y == 1:
        X(qubits[1])
    h(qubits[0])
    ctrl(qubits[0], X, qubits[1])
    return qubits

@code_ket
def pauli_prepare(basis : Union[x, y, z], q : quant, state : int = +1):
    """Prepares qubits in the +1 or -1 eigenstate of a given Pauli operator."""

    if state == -1:
        x(q)
    elif state == 1:
        pass
    else:
        raise ValueError('param state must be +1 or -1.')

    if basis == x:
        h(q)
    elif basis == y:
        h(q)
        s(q)
    elif basis == z:
        pass
    else:
        raise ValueError('param basis must be x, y, or z.')

def pauli_measure(basis : Union[x, y, z], q : quant):
    """Pauli measurement."""

    adj(pauli_prepare, basis, q)
    return measure(q)

@code_ket
def measure_free(q : quant) -> future:
    """Measure and free a quant."""
    res = measure(q)
    for i in q:
        m = measure(i)
        if m:
            x(i) 
    q.free()
    return res

def within(around, apply):
    """Applay around(); apply(); adj(around)."""
    around()
    apply()
    adj(around)

@code_ket
def x_not_mask(q : quant, mask : List[int]):
    """Apply Pauli X gates follwing a bit mask (apply on 0)."""
    
    for i, b in zip(mask, q):
        if i == 0:
            x(b)

@code_ket
def x_mask(q : quant, mask : List[int]):
    """Apply Pauli X gates follwing a bit mask (applay on 1)."""
    
    for i, b in zip(mask, q):
        if i:
            x(b)

def x_int(q : quant, int_mask : int):
    """Apply Pauli X gates follwing a int mask."""
    
    mask = [int(i) for i in ('{0:0'+str(len(q))+'b}').format(int_mask)]
    x_mask(q, mask)
 
def ctrl_mask(c : quant, mask : List[int], func, *args, **kwargs):
    """Applay a quantum operation if the qubits of control matchs the mask."""

    within(lambda : x_not_mask(c, mask), lambda : ctrl(c, func, *args, **kwargs))

def ctrl_int(c : quant, int_mask : int, func, *args, **kwargs):
    """Applay a quantum operation if the qubits of control matchs the integer value."""
    
    mask = [int(i) for i in ('{0:0'+str(len(c))+'b}').format(int_mask)]
    ctrl_mask(c, mask, func, *args, **kwargs)

def increment(q):
    """Add 1 to the superposition, 'q += 1'. """

    if len(q) > 1:
        ctrl(q[-1], increment, q[:-1])
    x(q[-1])

def dump_matrix(u, size : int) -> List[List[complex]]:
    """Get the matrix of a quantum operation."""

    mat = [[0 for _ in range(2**size)] for _ in range(2**size)]
    with run():
        i = quant(size)
        j = quant(size)
        h(j)
        cnot(j, i)
        ret = u(i)
        d = dump(j|i)
        exec_quantum()
    
    for state in d.get_states():
        j = state >> size
        i = state & ((1 << size)-1)
        mat[i][j] = d.amplitude(state)[0]/(1/sqrt(2**size))

    if ret != None:
        return mat, ret
    else:
        return mat
     