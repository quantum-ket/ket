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

from ... import *
from ... import code_ket
from math import sqrt, asin
from typing import Union

@code_ket
def bell(x : int = 0, y : int = 0, qubits : quant = None) -> quant:
    r"""Bell state preparation
    
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
    H(qubits[0])
    ctrl(qubits[0], X, qubits[1])
    return qubits

def ghz(qubits : Union[quant, int]) -> quant:
    """GHZ state preparation"""
    if type(qubits) == int:
        qubits = quant(qubits)
    
    ctrl(H(qubits[0]), X, qubits[1:])

    return qubits

def w(qubits : Union[quant, int]) -> quant:
    """W state preparation"""
    if type(qubits) == int:
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

@code_ket
def pauli(basis : Union[X, Y, Z], q : quant, state : int = +1) -> quant:
    """Prepares qubits in the +1 or -1 eigenstate of a given Pauli operator."""

    if state == -1:
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





