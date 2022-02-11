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

from ..libket import X as _X, Y as _Y, Z as _Z, H as _H, S as _S, SD as _SD, T as _T, TD as _TD
from ..libket import phase as _phase, RX as _RX, RY as _RY, RZ as _RZ 
from ..libket import quant
from ..standard import ctrl, around
from .quantum_gate import quantum_gate
from math import pi

__all__ = ['I', 'X', 'Y', 'Z', 'H', 'S', 'SD', 'T', 'TD', 'phase', 'RX', 'RY', 'RZ', 'cnot', 'swap', 'RXX', 'RYY', 'RZZ']

I = quantum_gate(
    name='Identity', 
    gate=lambda q : q, 
    doc={
        'func' : '``I(q)``', 
        'matrix' : r":math:`\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`",
        'effect' : r":math:`\begin{matrix} I\left|0\right> = & \left|0\right> \\ I\left|1\right> = & \left|1\right> \end{matrix}`"
    }
)

X = quantum_gate(
    name='Pauli-X', 
    gate=_X, 
    doc={
        'func' : '``X(q)``', 
        'matrix' : r":math:`\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}`",
        'effect' : r":math:`\begin{matrix} X\left|0\right> = & \left|1\right> \\ X\left|1\right> = & \left|0\right> \end{matrix}`"
    }
)

Y = quantum_gate(
    name='Pauli-Y', 
    gate=_Y, 
    doc={
        'func' : '``Y(q)``',
        'matrix' : r":math:`\begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}`",
        'effect' : r":math:`\begin{matrix} Y\left|0\right> = & i\left|1\right> \\ Y\left|1\right> =& -i\left|0\right> \end{matrix}`"
    }
)

Z = quantum_gate(
    name='Pauli-Z', 
    gate=_Z, 
    doc={
        'func' : '``Z(q)``',
        'matrix' : r":math:`\begin{bmatrix}  1 & 0 \\ 0 & -1 \end{bmatrix}`",
        'effect' : r":math:`\begin{matrix} Z\left|0\right> = & \left|0\right> \\ Z\left|1\right> = & -\left|1\right> \end{matrix}`"
    }
)

H = quantum_gate(
    name='Hadamard',
    gate=_H,
    doc={
        'func' : '``H(q)``',
        'matrix' : r":math:`\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}`",
        'effect' : r":math:`\begin{matrix} H\left|0\right> = & \frac{\left|0\right>+\left|1\right>}{\sqrt{2}} = & \left|+\right> \\ H\left|1\right> = & \frac{\left|0\right>-\left|1\right>}{\sqrt{2}} = & \left|-\right> \\ H\left|+\right> = & \left|0\right> \\ H\left|-\right> = & \left|1\right> \\ \end{matrix}`"
    }
)

S = quantum_gate(
    name='S',
    gate=_S,
    doc={
        'func' : '``S(q)``',
        'matrix' : r":math:`\begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}`",
        'effect' : r":math:`\begin{matrix}  S\left|0\right> = & \left|0\right> \\  S\left|1\right> = & i\left|1\right> \end{matrix}`" 
    }
)

SD = quantum_gate(
    name='S Dagger',
    gate=_SD,
    doc={
        'func' : '``SD(q)``',
        'matrix' : r":math:`\begin{bmatrix} 1 & 0 \\ 0 & -i \end{bmatrix}`",
        'effect' : r":math:`\begin{matrix} S^\dagger\left|0\right> = & \left|0\right> \\ S^\dagger\left|1\right> = & -i\left|1\right> \end{matrix}`"
    }
)

T = quantum_gate(
    name='T',
    gate=_T,
    doc={
        'func' : '``T(q)``',
        'matrix' : r":math:`\begin{bmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{bmatrix}`",
        'effect' : r":math:`\begin{matrix} T^\dagger\left|0\right> = & \left|0\right> \\ T^\dagger\left|1\right> = & \frac{1+i}{\sqrt{2}}\left|1\right> \end{matrix}`" 
    }
)


TD = quantum_gate(
    name='T Dagger',
    gate=_TD,
    doc={
        'func' : '``TD(q)``',
        'matrix' : r":math:`\begin{bmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{bmatrix}`",
        'effect' : r":math:`\begin{matrix} T^\dagger\left|0\right> = & \left|0\right> \\ T^\dagger\left|1\right> = & \frac{1-i}{\sqrt{2}}\left|1\right> \end{matrix}`" 
    }
)

phase = quantum_gate(
    name='Phase',
    gate=_phase,
    c_args=1,
    doc={
        'func' : '``phase(λ, q)``',
        'matrix' : r":math:`\begin{bmatrix} 1 & 0 \\ 0 & e^{i\lambda} \end{bmatrix}`",
        'effect' : r":math:`\begin{matrix} P\left|0\right> = & \left|0\right> \\ P\left|1\right> =& e^{i\lambda}\left|1\right> \end{matrix}`"
    }
) 

RX = quantum_gate(
    name='X-axis Rotation',
    gate=_RX,
    c_args=1,
    doc={
        'func' : '``RX(θ, q)``',
        'matrix' : r":math:`\begin{bmatrix} \cos{\frac{\theta}{2}} & -i\sin{\frac{\theta}{2}} \\ -i\sin{\frac{\theta}{2}} & \cos{\frac{\theta}{2}} \end{bmatrix}`",
        'effect' : r":math:`\begin{matrix} RX\left|0\right> = & \cos\frac{\theta}{2}\left|0\right> -i\sin\frac{\theta}{2}\left|1\right> \\ RX\left|1\right> =& -i\sin\frac{\theta}{2}\left|0\right> + \cos\frac{\theta}{2}\left|1\right> \end{matrix}`"
    }
)

RY = quantum_gate(
    name='Y-axis Rotation',
    gate=_RY,
    c_args=1,
    doc={
        'func' : '``RY(θ, q)``',
        'matrix' : r":math:`\begin{bmatrix} \cos{\frac{\theta}{2}} & -\sin{\frac{\theta}{2}} \\ \sin{\frac{\theta}{2}} & \cos{\frac{\theta}{2}} \end{bmatrix}`",
        'effect' : r":math:`\begin{matrix} RY\left|0\right> = & \cos{\theta\over2}\left|0\right> + \sin\frac{\theta}{2}\left|1\right> \\ RY\left|1\right> =& -\sin\frac{\theta}{2}\left|0\right> + \cos\frac{\theta}{2}\left|1\right> \end{matrix}`"
    }
)

RZ = quantum_gate(
    name='Z-axis Rotation',
    gate=_RZ,
    c_args=1,
    doc={
        'func' : '``RZ(θ, q)``',
        'matrix' : r":math:`\begin{bmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{bmatrix}`",      
        'effect' : r":math:`\begin{matrix} RZ\left|0\right> = & e^{-i\theta/2}\left|0\right> \\ RZ\left|1\right> =& e^{i\theta/2}\left|1\right> \end{matrix}`"
        }
)

def _cnot(c, t):
    for i, j in zip(c, t):
        ctrl(i, X, j)

cnot = quantum_gate(
    name='Controlled-NOT',
    gate=_cnot,
    q_args=2,
    doc={
        'func' : '``cnot(c, t)``',
        'matrix' : r":math:`\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}`"
    }

)
 
def _swap(a, b):
    _cnot(a, b)
    _cnot(b, a)
    _cnot(a, b)

swap = quantum_gate(
    name='SWAP',
    gate=_swap,
    q_args=2,
    doc={
        'func' : '``swap(a, b)``',
        'matrix' : r":math:`\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}`"
    }

)

def _RXX(theta, a, b):
    for qa, qb in zip(a, b):
        with around(cnot(H, H), qa, qb):
            RZ(theta, qb)

RXX = quantum_gate(
    name='XX-axis Rotation',
    gate=_RXX,
    c_args=1,
    q_args=2,
    doc={
        'func' : '``RXX(θ, a, b)``',
        'matrix' : r":math:`\begin{bmatrix} \cos\frac{\theta}{2} & 0 & 0 & -i\sin\frac{\theta}{2} \\ 0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\ 0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\ -i\sin\frac{\theta}{2} & 0 & 0 & \cos\frac{\theta}{2} \end{bmatrix}`"
    }
)

def _RYY(theta, a, b):
    for qa, qb in zip(a, b):
        with around(cnot(RX(pi/2), RX(pi/2)), qa, qb):
            RZ(theta, qb)

RYY = quantum_gate(
    name='YY-axis Rotation',
    gate=_RYY,
    c_args=1,
    q_args=2,
    doc={
        'func' : '``RYY(θ, a, b)``',
        'matrix' : r":math:`\begin{bmatrix} \cos\frac{\theta}{2} & 0 & 0 & i\sin\frac{\theta}{2} \\ 0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\ 0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\ i\sin\frac{\theta}{2} & 0 & 0 & \cos\frac{\theta}{2} \end{bmatrix}`"
    }
)

def _RZZ(theta, a, b):
    for qa, qb in zip(a, b):
        with around(cnot, qa, qb):
            RZ(theta, qb)

RZZ = quantum_gate(
    name='ZZ-axis Rotation',
    gate=_RZZ,
    c_args=1,
    q_args=2,
    doc={
        'func' : '``RZZ(θ, a, b)``',
        'matrix' : r":math:`\begin{bmatrix} e^{-i \frac{\theta}{2}} & 0 & 0 & 0 \\ 0 & e^{i \frac{\theta}{2}} & 0 & 0 \\ 0 & 0 & e^{i \frac{\theta}{2}} & 0 \\ 0 & 0 & 0 & e^{-i \frac{\theta}{2}} \end{bmatrix}`"
    }
)
            