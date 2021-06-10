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

from ..ket import X as _X, Y as _Y, Z as _Z, H as _H, S as _S, SD as _SD, T as _T, TD as _TD
from ..ket import phase as _phase, RX as _RX, RY as _RY, RZ as _RZ 
from ..types import quant
from ..standard import ctrl
from typing import Callable, Optional

__all__ = ['I', 'X', 'Y', 'Z', 'H', 'S', 'SD', 'T', 'TD', 'phase', 'RX', 'RY', 'RZ', 'cnot', 'swap']

def X(q : quant) -> quant:
    r""" Pauli-X gate (:math:`\sigma_x`)

    Apply a single-qubit Pauli-X gate on every qubit of ``q``.

    **Matrix representation:**

    .. math::

        X = \begin{bmatrix} 
                0 & 1 \\
                1 & 0 
            \end{bmatrix}

    **Effect:**

    .. math::

        \begin{matrix} 
            X\left|0\right> = & \left|1\right> \\
            X\left|1\right> = & \left|0\right>
        \end{matrix}

    :param q: Input qubits.
    """
    _X(q)
    return q

def Y(q : quant) -> quant:
    r"""Pauli-Y gate (:math:`\sigma_y`)

    Apply a single-qubit Pauli-Y gate on every qubit of ``q``.

    **Matrix representation:**

    .. math::

        Y = \begin{bmatrix} 
                   0 & -i \\
                   i & 0 
            \end{bmatrix}

    **Effect:**

    .. math::

        \begin{matrix} 
            Y\left|0\right> = & -i\left|1\right> \\
            Y\left|1\right> =& i\left|1\right>
        \end{matrix}

    :param q: Input qubits.
    """
    _Y(q)
    return q

def Z(q : quant) -> quant:
    r"""Pauli-Z gate (:math:`\sigma_z`)

    Apply a single-qubit Pauli-Y gate on every qubit of ``q``.

    **Matrix representation:**

    .. math::

        Z = \begin{bmatrix} 
                1 & 0 \\
                0 & -1 
            \end{bmatrix}

    **Effect:**

    .. math::

        \begin{matrix} 
            Z\left|0\right> = & \left|0\right> \\ 
            Z\left|1\right> = & -\left|1\right>
        \end{matrix}

    :param q: Input qubits.
    """
    _Z(q)
    return q

def H(q : quant) -> quant:
    r"""Hadamard gate

    Apply a single-qubit Hadamard gate on every qubit of ``q``.

    **Matrix representation:**

    .. math::

        H = \frac{1}{\sqrt{2}}\begin{bmatrix} 
                                    1 & 1 \\
                                    1 & -1 
                              \end{bmatrix}

    **Effect:**

    .. math::

        \begin{matrix} 
            H\left|0\right> = & \frac{\left|0\right>+\left|1\right>}{\sqrt{2}} = & \left|+\right> \\
            H\left|1\right> = & \frac{\left|0\right>-\left|1\right>}{\sqrt{2}} = & \left|-\right> \\
            H\left|+\right> = & \left|0\right> \\
            H\left|-\right> = & \left|1\right> \\
        \end{matrix}

    :param q: Input qubits.
    """
    _H(q)
    return q

def S(q : quant) -> quant:
    r"""S gate 

    Apply a single-qubit S gate on every qubit of ``q``.

    **Matrix representation:**

    .. math::

        S = \begin{bmatrix} 
                1 & 0 \\
                0 & i 
            \end{bmatrix}

    **Effect:**

    .. math::

        \begin{matrix} 
            S\left|0\right> = & \left|0\right> \\
            S\left|1\right> = & i\left|1\right> 
        \end{matrix}

    :param q: Input qubits.
    """
    _S(q)
    return q

def SD(q : quant) -> quant:
    r"""S-dagger gate (:math:`S^\dagger`)

    Apply a single-qubit S-dagger gate on every qubit of ``q``.

    **Matrix representation:**

    .. math::

        S^\dagger = \begin{bmatrix} 
                        1 & 0 \\
                        0 & -i 
                    \end{bmatrix}

    **Effect:**

    .. math::

        \begin{matrix} 
                 S^\dagger\left|0\right> = & \left|0\right> \\
                 S^\dagger\left|1\right> = & -i\left|1\right> 
             \end{matrix}

    :param q: Input qubits.
    """
    _SD(q)
    return q

def T(q : quant) -> quant:
    r"""T gate

    Apply a single-qubit T gate on every qubit of ``q``.

    **Matrix representation:**

    .. math::

         T = \begin{bmatrix} 
                    1 & 0 \\
                    0 & e^{i\pi/4} 
             \end{bmatrix}

    **Effect:**

    .. math::

        \begin{matrix} 
            T\left|0\right> = & \left|0\right> \\
            T\left|1\right> = & \frac{1+i}{\sqrt{2}}\left|1\right> \end{matrix}

    :param q: Input qubits.
    """
    _T(q)
    return q

def TD(q : quant) -> quant:
    r"""T-dagger gate (:math:`T^\dagger`)

    Apply a single-qubit T-dagger gate on every qubit of ``q``.

    **Matrix representation:**

    .. math::

         T^\dagger = \begin{bmatrix} 
                        1 & 0 \\
                        0 & e^{-i\pi/4}
                     \end{bmatrix}

    **Effect:**

    .. math::

        \begin{matrix} 
                T^\dagger\left|0\right> = & \left|0\right> \\
                T^\dagger\left|1\right> = & \frac{1-i}{\sqrt{2}}\left|1\right> \end{matrix}

    :param q: Input qubits.
    """
    _TD(q)
    return q

def phase(lamb : float, q : Optional[quant] = None) -> Callable[[quant], quant] | quant:
    r"""Phase gate

    Apply a relative phase of :math:`e^{i\lambda}` on every qubit of ``q``.

    If ``q`` is not provided, return a new gate: ``phase(lamb)(q)``
    :math:`\equiv` ``phase(lamb, q)``. 


    **Matrix representation:**

    .. math::

        P(\lambda) = \begin{bmatrix}
                        1 & 0 \\
                        0 & e^{i\lambda}
                     \end{bmatrix}

    **Effect:**

    .. math::

        \begin{matrix}
                P\left|0\right> = & \left|0\right> \\
                P\left|1\right> =& e^{i\lambda}\left|1\right> 
        \end{matrix}

    :param q: Input qubits.
    :param lamb: Relative phase :math:`\lambda`.
    """
    if q is not None:
       _phase(lamb, q)
       return q
    else:
        def __phase(q):
            _phase(lamb, q)
            return q
        return __phase


def RX(theta : float, q : Optional[quant] = None) -> Callable[[quant], quant] | quant:
    r"""X-axis rotation gate

    Apply a rotation of :math:`\theta` about the X-axis on every qubit of ``q``.

    If ``q`` is not provided, return a new gate: ``RX(theta)(q)``
    :math:`\equiv` ``RX(theta, q)``. 

    **Matrix representation:**

    .. math::

         RX(\theta) = \begin{bmatrix} 
                        \cos{\frac{\theta}{2}} & -i\sin{\frac{\theta}{2}} \\
                        -i\sin{\frac{\theta}{2}} & \cos{\frac{\theta}{2}} 
                      \end{bmatrix}

    **Effect:**

    .. math::

        \begin{matrix}
                RX\left|0\right> = & \cos\frac{\theta}{2}\left|0\right> -i\sin\frac{\theta}{2}\left|1\right> \\
                RX\left|1\right> =& -i\sin\frac{\theta}{2}\left|0\right> + \cos\frac{\theta}{2}\left|1\right>
        \end{matrix}

    :param q: Input qubits.
    :param theta: Rotation angle :math:`\theta`.
    """
    if q is not None:
       _RX(theta, q)
       return q
    else:
        def __RX(q):
            _RX(theta, q)
            return q
        return __RX

def RY(theta : float, q : Optional[quant] = None) -> Callable[[quant], quant] | quant:
    r"""Y-axis rotation gate

    Apply a rotation of :math:`\theta` about the Y-axis on every qubit of ``q``.

    If ``q`` is not provided, return a new gate: ``RY(theta)(q)``
    :math:`\equiv` ``RY(theta, q)``. 

    **Matrix representation:**

    .. math::

         RY(\theta) = \begin{bmatrix} 
                        \cos{\frac{\theta}{2}} & -\sin{\frac{\theta}{2}} \\
                        \sin{\frac{\theta}{2}} & \cos{\frac{\theta}{2}} 
                      \end{bmatrix}

    **Effect:**

    .. math::

        \begin{matrix}
            RY\left|0\right> = & \cos{\theta\over2}\left|0\right> -i\sin\frac{\theta}{2}\left|1\right> \\
            RY\left|1\right> =& -\sin\frac{\theta}{2}\left|0\right> + \cos\frac{\theta}{2}\left|1\right> 
        \end{matrix}

    :param q: Input qubits.
    :param theta: Rotation angle :math:`\theta`.
    """
    if q is not None:
       _RY(theta, q)
       return q
    else:
        def __RY(q):
            _RY(theta, q)
            return q
        return __RY

def RZ(theta : float, q : Optional[quant] = None) -> Callable[[quant], quant] | quant:
    r"""Z-axis rotation gate

    Apply a rotation of :math:`\theta` about the Z-axis on every qubit of ``q``.

    If ``q`` is not provided, return a new gate: ``RZ(theta)(q)``
    :math:`\equiv` ``RZ(theta, q)``. 

    **Matrix representation:**

    .. math::

         RZ(\theta) =  \begin{bmatrix} 
                            e^{-i\theta/2} & 0 \\
                            0 & e^{i\theta/2} 
                       \end{bmatrix}

    **Effect:**

    .. math::

        \begin{matrix}
            RZ\left|0\right> = & e^{-i\theta/2}\left|0\right> \\ 
            RZ\left|1\right> =& e^{i\theta/2}\left|1\right> 
        \end{matrix}

    :param q: Input qubits.
    :param theta: Rotation angle :math:`\theta`.
    """
    if q is not None:
       _RZ(theta, q)
       return q
    else:
        def __RZ(q):
            _RZ(theta, q)
            return q
        return __RZ


def I(q : quant) -> quant:
    r"""Identity gate 

    Apply an identity gate on every qubit of ``q``.

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

    :param q: Input qubits.
    """
    return q
 
def cnot(c : quant, t : quant) -> tuple[quant, quant]:
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

    :param c: Control qubits.
    :param t: Target qubits.
    """
    
    for i, j in zip(c, t):
        ctrl(i, X, j)
    
    return c, t

def swap(a : quant, b : quant) -> tuple[quant, quant]:
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

    :param a: Input qubits.
    :param b: Input qubits.
    """
    cnot(a, b)
    cnot(b, a)
    cnot(a, b)

    return a, b
