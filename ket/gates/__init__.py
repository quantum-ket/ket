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

from ..ket import X, Y, Z, H, S, SD, T, TD, quant
from ..ket import phase as _phase, RX as _RX, RY as _RY, RZ as _RZ 
from ..standard import ctrl
from typing import Callable, Tuple, Union, Optional

__all__ = ['I', 'X', 'Y', 'Z', 'H', 'S', 'SD', 'T', 'TD', 'phase', 'RX', 'RY', 'RZ', 'cnot', 'swap']

def RX(theta : float, q : Optional[quant] = None) -> Union[Callable, quant]:
    if q is not None:
        return _RX(theta, q)
    else:
        return lambda q : _RX(theta, q)

def RY(theta : float, q : Optional[quant] = None) -> Union[Callable, quant]:
    if q is not None:
        return _RY(theta, q)
    else:
        return lambda q : _RY(theta, q)

def RZ(theta : float, q : Optional[quant] = None) -> Union[Callable, quant]:
    if q is not None:
        return _RZ(theta, q)
    else:
        return lambda q : _RZ(theta, q)

def phase(_lambda : float, q : Optional[quant] = None) -> Union[Callable, quant]:
    if q is not None:
        return _phase(_lambda, q)
    else:
        return lambda q : _phase(_lambda, q)

RX.__doc__ = _RX.__doc__
RY.__doc__ = _RY.__doc__
RZ.__doc__ = _RZ.__doc__
phase.__doc__ = _phase.__doc__

def I(q : quant) -> quant:
    r"""Identity gate 

    Apply an identity gate on every qubit of q.

    Note:
        This quantum gate does nothing! 

    **Matrix representation:**

    .. math::

        I = \begin{bmatrix} 
                     1 & 0 \\
                     0 & 1 
                 \end{bmatrix}

    **Effect:**

    .. math::

        \begin{matrix} 
                 X\left|0\right> = & \left|0\right> \\
                 X\left|1\right> = & \left|1\right>
             \end{matrix}

    :type q: :py:class:`quant`
    :param q: input qubits
    """
    return q
 
def cnot(c : quant, t : quant) -> Tuple[quant, quant]:
    r"""Controlled-NOT
    
    Apply a CNOT gate between the qubits of c and t.

    **Matrix representation:**

    .. math::

        CNOT = \begin{bmatrix} 
                     1 & 0 & 0 & 0\\
                     0 & 1 & 0 & 0\\
                     0 & 0 & 0 & 1\\
                     0 & 0 & 1 & 0
                 \end{bmatrix}

    **Effect:**

    .. math::

        \begin{matrix} 
                 CNOT\left|c t\right> = & \left|c (c\oplus t)\right> 
        \end{matrix}

    :param c: control qubits
    :param t: target qubits
    """
    
    for i, j in zip(c, t):
        ctrl(i, X, j)
    
    return c, t

def swap(a : quant, b : quant) -> Tuple[quant, quant]:
    r"""SWAP gate
    
    Swap the qubits of a and b.

    **Matrix representation:**

    .. math::

        CNOT = \begin{bmatrix} 
                     1 & 0 & 0 & 0\\
                     0 & 0 & 1 & 0\\
                     0 & 1 & 0 & 0\\
                     0 & 0 & 0 & 1
                 \end{bmatrix}

    **Effect:**

    .. math::

        \begin{matrix} 
                 CNOT\left|a b\right> = & \left|b a\right> 
        \end{matrix}

    :param a: input qubits
    :param b: input qubits
    """
    cnot(a, b)
    cnot(b, a)
    cnot(a, b)

    return a, b
