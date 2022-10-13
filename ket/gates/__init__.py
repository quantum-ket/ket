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

from ..base import (base_H, base_phase, base_RX, base_RY, base_RZ,
                    base_S, base_SD, base_T, base_TD, base_X, base_Y, base_Z)
from .quantum_gate import QuantumGate
from ..standard.ctrl import base_flipc
from .base_gates import base_cnot, base_phase_on, base_RXX, base_RZZ, base_RYY, base_swap

__all__ = ['I', 'X', 'Y', 'Z', 'H', 'S', 'SD', 'T', 'TD', 'phase', 'RX',
           'RY', 'RZ', 'cnot', 'swap', 'RXX', 'RYY', 'RZZ', 'flipc', 'phase_on']

# pylint: disable=line-too-long

I = QuantumGate(  # noqa: E741
    name='Identity',
    gate=lambda q: q,
    doc={
        'func': '``I(q)``',
        'matrix': r":math:`\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`",
        'effect': r":math:`\begin{matrix} I\left|0\right> = & \left|0\right> \\ I\left|1\right> = & \left|1\right> \end{matrix}`"
    }
)

X = QuantumGate(
    name='Pauli-X',
    gate=base_X,
    doc={
        'func': '``X(q)``',
        'matrix': r":math:`\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}`",
        'effect': r":math:`\begin{matrix} X\left|0\right> = & \left|1\right> \\ X\left|1\right> = & \left|0\right> \end{matrix}`"
    }
)

Y = QuantumGate(
    name='Pauli-Y',
    gate=base_Y,
    doc={
        'func': '``Y(q)``',
        'matrix': r":math:`\begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}`",
        'effect': r":math:`\begin{matrix} Y\left|0\right> = & i\left|1\right> \\ Y\left|1\right> =& -i\left|0\right> \end{matrix}`"
    }
)

Z = QuantumGate(
    name='Pauli-Z',
    gate=base_Z,
    doc={
        'func': '``Z(q)``',
        'matrix': r":math:`\begin{bmatrix}  1 & 0 \\ 0 & -1 \end{bmatrix}`",
        'effect': r":math:`\begin{matrix} Z\left|0\right> = & \left|0\right> \\ Z\left|1\right> = & -\left|1\right> \end{matrix}`"
    }
)

H = QuantumGate(
    name='Hadamard',
    gate=base_H,
    doc={
        'func': '``H(q)``',
        'matrix': r":math:`\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}`",
        'effect': r":math:`\begin{matrix} H\left|0\right> = & \frac{\left|0\right>+\left|1\right>}{\sqrt{2}} = & \left|+\right> \\ H\left|1\right> = & \frac{\left|0\right>-\left|1\right>}{\sqrt{2}} = & \left|-\right> \\ H\left|+\right> = & \left|0\right> \\ H\left|-\right> = & \left|1\right> \\ \end{matrix}`"
    }
)

S = QuantumGate(
    name='S',
    gate=base_S,
    doc={
        'func': '``S(q)``',
        'matrix': r":math:`\begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}`",
        'effect': r":math:`\begin{matrix}  S\left|0\right> = & \left|0\right> \\  S\left|1\right> = & i\left|1\right> \end{matrix}`"
    }
)

SD = QuantumGate(
    name='S Dagger',
    gate=base_SD,
    doc={
        'func': '``SD(q)``',
        'matrix': r":math:`\begin{bmatrix} 1 & 0 \\ 0 & -i \end{bmatrix}`",
        'effect': r":math:`\begin{matrix} S^\dagger\left|0\right> = & \left|0\right> \\ S^\dagger\left|1\right> = & -i\left|1\right> \end{matrix}`"
    }
)

T = QuantumGate(
    name='T',
    gate=base_T,
    doc={
        'func': '``T(q)``',
        'matrix': r":math:`\begin{bmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{bmatrix}`",
        'effect': r":math:`\begin{matrix} T^\dagger\left|0\right> = & \left|0\right> \\ T^\dagger\left|1\right> = & \frac{1+i}{\sqrt{2}}\left|1\right> \end{matrix}`"
    }
)


TD = QuantumGate(
    name='T Dagger',
    gate=base_TD,
    doc={
        'func': '``TD(q)``',
        'matrix': r":math:`\begin{bmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{bmatrix}`",
        'effect': r":math:`\begin{matrix} T^\dagger\left|0\right> = & \left|0\right> \\ T^\dagger\left|1\right> = & \frac{1-i}{\sqrt{2}}\left|1\right> \end{matrix}`"
    }
)

phase = QuantumGate(
    name='Phase',
    gate=base_phase,
    c_args=1,
    doc={
        'func': '``phase(λ, q)``',
        'matrix': r":math:`\begin{bmatrix} 1 & 0 \\ 0 & e^{i\lambda} \end{bmatrix}`",
        'effect': r":math:`\begin{matrix} P\left|0\right> = & \left|0\right> \\ P\left|1\right> =& e^{i\lambda}\left|1\right> \end{matrix}`"
    }
)

RX = QuantumGate(
    name='X-axis Rotation',
    gate=base_RX,
    c_args=1,
    doc={
        'func': '``RX(θ, q)``',
        'matrix': r":math:`\begin{bmatrix} \cos{\frac{\theta}{2}} & -i\sin{\frac{\theta}{2}} \\ -i\sin{\frac{\theta}{2}} & \cos{\frac{\theta}{2}} \end{bmatrix}`",
        'effect': r":math:`\begin{matrix} RX\left|0\right> = & \cos\frac{\theta}{2}\left|0\right> -i\sin\frac{\theta}{2}\left|1\right> \\ RX\left|1\right> =& -i\sin\frac{\theta}{2}\left|0\right> + \cos\frac{\theta}{2}\left|1\right> \end{matrix}`"
    }
)

RY = QuantumGate(
    name='Y-axis Rotation',
    gate=base_RY,
    c_args=1,
    doc={
        'func': '``RY(θ, q)``',
        'matrix': r":math:`\begin{bmatrix} \cos{\frac{\theta}{2}} & -\sin{\frac{\theta}{2}} \\ \sin{\frac{\theta}{2}} & \cos{\frac{\theta}{2}} \end{bmatrix}`",
        'effect': r":math:`\begin{matrix} RY\left|0\right> = & \cos{\theta\over2}\left|0\right> + \sin\frac{\theta}{2}\left|1\right> \\ RY\left|1\right> =& -\sin\frac{\theta}{2}\left|0\right> + \cos\frac{\theta}{2}\left|1\right> \end{matrix}`"
    }
)

RZ = QuantumGate(
    name='Z-axis Rotation',
    gate=base_RZ,
    c_args=1,
    doc={
        'func': '``RZ(θ, q)``',
        'matrix': r":math:`\begin{bmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{bmatrix}`",
        'effect': r":math:`\begin{matrix} RZ\left|0\right> = & e^{-i\theta/2}\left|0\right> \\ RZ\left|1\right> =& e^{i\theta/2}\left|1\right> \end{matrix}`"
    }
)


cnot = QuantumGate(
    name='Controlled-NOT',
    gate=base_cnot,
    q_args=2,
    doc={
        'func': '``cnot(c, t)``',
        'matrix': r":math:`\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}`"
    }

)

swap = QuantumGate(
    name='SWAP',
    gate=base_swap,
    q_args=2,
    doc={
        'func': '``swap(a, b)``',
        'matrix': r":math:`\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}`"
    }

)

RXX = QuantumGate(
    name='XX-axis Rotation',
    gate=base_RXX,
    c_args=1,
    q_args=2,
    doc={
        'func': '``RXX(θ, a, b)``',
        'matrix': r":math:`\begin{bmatrix} \cos\frac{\theta}{2} & 0 & 0 & -i\sin\frac{\theta}{2} \\ 0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\ 0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\ -i\sin\frac{\theta}{2} & 0 & 0 & \cos\frac{\theta}{2} \end{bmatrix}`"
    }
)

RYY = QuantumGate(
    name='YY-axis Rotation',
    gate=base_RYY,
    c_args=1,
    q_args=2,
    doc={
        'func': '``RYY(θ, a, b)``',
        'matrix': r":math:`\begin{bmatrix} \cos\frac{\theta}{2} & 0 & 0 & i\sin\frac{\theta}{2} \\ 0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\ 0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\ i\sin\frac{\theta}{2} & 0 & 0 & \cos\frac{\theta}{2} \end{bmatrix}`"
    }
)

RZZ = QuantumGate(
    name='ZZ-axis Rotation',
    gate=base_RZZ,
    c_args=1,
    q_args=2,
    doc={
        'func': '``RZZ(θ, a, b)``',
        'matrix': r":math:`\begin{bmatrix} e^{-i \frac{\theta}{2}} & 0 & 0 & 0 \\ 0 & e^{i \frac{\theta}{2}} & 0 & 0 \\ 0 & 0 & e^{i \frac{\theta}{2}} & 0 \\ 0 & 0 & 0 & e^{-i \frac{\theta}{2}} \end{bmatrix}`"
    }
)

flipc = QuantumGate(
    name='Flip to Control State',
    gate=base_flipc,
    c_args=1,
    q_args=None,
    doc={
        'func': '``flipc(state, q)``',
        'matrix': r":math:`\bigotimes_{\text{bit }\in\text{ state}}\begin{cases}X,&\text{for bit }=0\\I,&\text{for bit }=1\end{cases}`"
    }
)

phase_on = QuantumGate(
    name='Phase on State',
    gate=base_phase_on,
    c_args=1,
    q_args=None,
    doc={
        'func': '``phase_on(state, q)``',
        'matrix': r":math:`\sum\delta_k\left|k\right>\left<k\right|,\text{where }\delta_k=\begin{cases}-1,&\text{for }k= \text{state}\\1,&\text{for }k\neq\text{state}\end{cases}`"
    }
)
