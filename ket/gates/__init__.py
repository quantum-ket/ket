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

from ..ket import x, y, z, h, s, sd, t, td, u1, u2, u3, rx, ry, rz, quant
from ..standard import ctrl

__all__ = ['i', 'x', 'y', 'z', 'h', 's', 'sd', 't', 'td', 'p', 'u1', 'u2', 'u3', 'rx', 'ry', 'rz',
           'I', 'X', 'Y', 'Z', 'H', 'S', 'SD', 'T', 'TD', 'P', 'U1', 'U2', 'U','U3', 'RX', 'RY', 'RZ',
           'cnot', 'swap']

def i(q : quant):
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
    return
 
def cnot(c : quant, t : quant):
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
        ctrl(i, x, j)

def swap(a : quant, b : quant):
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

I  = i            
X  = x 
Y  = y
Z  = z
H  = h
S  = s
SD = sd
T  = t
TD = td
p = u1
P = u1
U1 = u1
U2 = u2
U = u3
U3 = u3
RX = rx
RY = ry
RZ = rz
