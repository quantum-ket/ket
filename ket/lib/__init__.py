# MIT License
# 
# Copyright (c) 2020 Evandro Chagas Ribeiro da Rosa <evandro.crr@posgrad.ufsc.br>
# Copyright (c) 2020 Rafael de Santiago <r.santiago@ufsc.br>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from .. import *
from ..code_ket import code_ket
from math import pi, sqrt
from typing import Union, List

def qft(q : quant, invert=True):
    """Apply a Quantum Fourier Transformation."""
    
    for i in range(len(q)):
        h(q[i])
        for j, m in zip(range(i+1, len(q)), [2*pi/2**m for m in range(2, len(q)+1)]):
            ctrl(q[j], u1, m, q[i])
    if invert:
        for i in range(len(q)//2):
            swap(q[i], q[len(q)-i-1])

def cnot(c : quant, t : quant):
    """Quantum-bitwise Controlled-NOT."""
    
    for i, j in zip(c, t):
        ctrl(i, x, j)

def swap(a : quant, b : quant):
    """Quantum-bitwise swap."""
   
    cnot(a, b)
    cnot(b, a)
    cnot(a, b)

@code_ket
def bell(aux0 : int, aux1 : int) -> quant:
    """Return two entangle qubits in the Bell state."""
   
    q = quant(2)
    if aux0 == 1:
        x(q[0])
    if aux1 == 1:
        x(q[1])
    h(q[0])
    ctrl(q[0], x, q[1])
    return q

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
        u(i)
        d = dump(j|i)
        exec_quantum()
    
    for state in d.get_states():
        j = state >> size
        i = state & ((1 << size)-1)
        mat[i][j] = d.amplitude(state)[0]/(1/sqrt(2**size))

    return mat
     