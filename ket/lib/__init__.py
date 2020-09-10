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
from math import pi

def qft(q : quant):
    """Apply a Quantum Fourier Transformation.
    
    note:
        The operation leaves the qubits in the inverse order.
    """
    
    lambd = lambda k : pi*k/2
    for i in range(len(q)):
        for j in range(i):
            ctrl(q[i], u1, lambd(i-j), q[j])
        h(q[i])

def cnot(c : quant, t : quant):
    """Quantum-bitwise Controlled-NOT."""
    for i, j in zip(c, t):
        ctrl(i, x, j)

def swap(a : quant, b : quant):
    """Quantum-bitwise swap."""
    cnot(a, b)
    cnot(b, a)
    cnot(a, b)

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

