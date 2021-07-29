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
from ..standard import ctrl, around
from typing import Callable, Optional, Iterable
from functools import reduce
from operator import add
from math import pi

__all__ = ['I', 'X', 'Y', 'Z', 'H', 'S', 'SD', 'T', 'TD', 'phase', 'RX', 'RY', 'RZ', 'cnot', 'swap', 'RXX', 'RYY', 'RZZ']

def X(q : quant | Iterable[quant]) -> quant:
    r""" Pauli-X gate (:math:`\sigma_x`)

    Apply a single-qubit Pauli-X gate on every qubit of ``q``.

    :Matrix Representation:

    .. math::

        X = \begin{bmatrix} 
                0 & 1 \\
                1 & 0 
            \end{bmatrix}

    :Effect:

    .. math::

        \begin{matrix} 
            X\left|0\right> = & \left|1\right> \\
            X\left|1\right> = & \left|0\right>
        \end{matrix}

    :param q: Input qubits.
    """
    q = reduce(add, q)
    _X(q)
    return q

def Y(q : quant | Iterable[quant]) -> quant:
    r"""Pauli-Y gate (:math:`\sigma_y`)

    Apply a single-qubit Pauli-Y gate on every qubit of ``q``.

    :Matrix Representation:

    .. math::

        Y = \begin{bmatrix} 
                   0 & -i \\
                   i & 0 
            \end{bmatrix}

    :Effect:

    .. math::

        \begin{matrix} 
            Y\left|0\right> = & -i\left|1\right> \\
            Y\left|1\right> =& i\left|1\right>
        \end{matrix}

    :param q: Input qubits.
    """
    q = reduce(add, q)
    _Y(q)
    return q

def Z(q : quant | Iterable[quant]) -> quant:
    r"""Pauli-Z gate (:math:`\sigma_z`)

    Apply a single-qubit Pauli-Y gate on every qubit of ``q``.

    :Matrix Representation:

    .. math::

        Z = \begin{bmatrix} 
                1 & 0 \\
                0 & -1 
            \end{bmatrix}

    :Effect:

    .. math::

        \begin{matrix} 
            Z\left|0\right> = & \left|0\right> \\ 
            Z\left|1\right> = & -\left|1\right>
        \end{matrix}

    :param q: Input qubits.
    """
    q = reduce(add, q)
    _Z(q)
    return q

def H(q : quant | Iterable[quant]) -> quant:
    r"""Hadamard gate

    Apply a single-qubit Hadamard gate on every qubit of ``q``.

    :Matrix Representation:

    .. math::

        H = \frac{1}{\sqrt{2}}\begin{bmatrix} 
                                    1 & 1 \\
                                    1 & -1 
                              \end{bmatrix}

    :Effect:

    .. math::

        \begin{matrix} 
            H\left|0\right> = & \frac{\left|0\right>+\left|1\right>}{\sqrt{2}} = & \left|+\right> \\
            H\left|1\right> = & \frac{\left|0\right>-\left|1\right>}{\sqrt{2}} = & \left|-\right> \\
            H\left|+\right> = & \left|0\right> \\
            H\left|-\right> = & \left|1\right> \\
        \end{matrix}

    :param q: Input qubits.
    """
    q = reduce(add, q)
    _H(q)
    return q

def S(q : quant | Iterable[quant]) -> quant:
    r"""S gate 

    Apply a single-qubit S gate on every qubit of ``q``.

    :Matrix Representation:

    .. math::

        S = \begin{bmatrix} 
                1 & 0 \\
                0 & i 
            \end{bmatrix}

    :Effect:

    .. math::

        \begin{matrix} 
            S\left|0\right> = & \left|0\right> \\
            S\left|1\right> = & i\left|1\right> 
        \end{matrix}

    :param q: Input qubits.
    """
    q = reduce(add, q)
    _S(q)
    return q

def SD(q : quant | Iterable[quant]) -> quant:
    r"""S-dagger gate (:math:`S^\dagger`)

    Apply a single-qubit S-dagger gate on every qubit of ``q``.
    
    ``SD`` :math:`\equiv` ``adj(S)``

    :Matrix Representation:

    .. math::

        S^\dagger = \begin{bmatrix} 
                        1 & 0 \\
                        0 & -i 
                    \end{bmatrix}

    :Effect:

    .. math::

        \begin{matrix} 
                 S^\dagger\left|0\right> = & \left|0\right> \\
                 S^\dagger\left|1\right> = & -i\left|1\right> 
             \end{matrix}

    :param q: Input qubits.
    """
    q = reduce(add, q)
    _SD(q)
    return q

def T(q : quant | Iterable[quant]) -> quant:
    r"""T gate

    Apply a single-qubit T gate on every qubit of ``q``.

    :Matrix Representation:

    .. math::

         T = \begin{bmatrix} 
                    1 & 0 \\
                    0 & e^{i\pi/4} 
             \end{bmatrix}

    :Effect:

    .. math::

        \begin{matrix} 
            T\left|0\right> = & \left|0\right> \\
            T\left|1\right> = & \frac{1+i}{\sqrt{2}}\left|1\right> \end{matrix}

    :param q: Input qubits.
    """
    q = reduce(add, q)
    _T(q)
    return q

def TD(q : quant | Iterable[quant]) -> quant:
    r"""T-dagger gate (:math:`T^\dagger`)

    Apply a single-qubit T-dagger gate on every qubit of ``q``.

    ``ST`` :math:`\equiv` ``adj(T)``

    :Matrix Representation:

    .. math::

         T^\dagger = \begin{bmatrix} 
                        1 & 0 \\
                        0 & e^{-i\pi/4}
                     \end{bmatrix}

    :Effect:

    .. math::

        \begin{matrix} 
                T^\dagger\left|0\right> = & \left|0\right> \\
                T^\dagger\left|1\right> = & \frac{1-i}{\sqrt{2}}\left|1\right> \end{matrix}

    :param q: Input qubits.
    """
    q = reduce(add, q)
    _TD(q)
    return q

def phase(lamb : float, q : Optional[quant | Iterable[quant]] = None) -> Callable[[quant], quant] | quant:
    r"""Phase gate

    Apply a relative phase of :math:`e^{i\lambda}` on every qubit of ``q``.

    If ``q`` is not provided, return a new gate: ``phase(lamb)(q)``
    :math:`\equiv` ``phase(lamb, q)``. 


    :Matrix Representation:

    .. math::

        P(\lambda) = \begin{bmatrix}
                        1 & 0 \\
                        0 & e^{i\lambda}
                     \end{bmatrix}

    :Effect:

    .. math::

        \begin{matrix}
                P\left|0\right> = & \left|0\right> \\
                P\left|1\right> =& e^{i\lambda}\left|1\right> 
        \end{matrix}

    :param q: Input qubits.
    :param lamb: Relative phase :math:`\lambda`.
    """
    def __phase(q):
        q = reduce(add, q)
        _phase(lamb, q)
        return q
      
    return __phase(q) if q is not None else __phase

def RX(theta : float, q : Optional[quant | Iterable[quant]] = None) -> Callable[[quant], quant] | quant:
    r"""X-axis rotation gate

    Apply a rotation of :math:`\theta` about the X-axis on every qubit of ``q``.

    If ``q`` is not provided, return a new gate: ``RX(theta)(q)``
    :math:`\equiv` ``RX(theta, q)``. 

    :Matrix Representation:

    .. math::

         RX(\theta) = \begin{bmatrix} 
                        \cos{\frac{\theta}{2}} & -i\sin{\frac{\theta}{2}} \\
                        -i\sin{\frac{\theta}{2}} & \cos{\frac{\theta}{2}} 
                      \end{bmatrix}

    :Effect:

    .. math::

        \begin{matrix}
                RX\left|0\right> = & \cos\frac{\theta}{2}\left|0\right> -i\sin\frac{\theta}{2}\left|1\right> \\
                RX\left|1\right> =& -i\sin\frac{\theta}{2}\left|0\right> + \cos\frac{\theta}{2}\left|1\right>
        \end{matrix}

    :param q: Input qubits.
    :param theta: Rotation angle :math:`\theta`.
    """
    def __RX(q):
        q = reduce(add, q)
        _RX(theta, q)
        return q
     
    return __RX(q) if q is not None else __RX

def RY(theta : float, q : Optional[quant | Iterable[quant]] = None) -> Callable[[quant], quant] | quant:
    r"""Y-axis rotation gate

    Apply a rotation of :math:`\theta` about the Y-axis on every qubit of ``q``.

    If ``q`` is not provided, return a new gate: ``RY(theta)(q)``
    :math:`\equiv` ``RY(theta, q)``. 

    :Matrix Representation:

    .. math::

         RY(\theta) = \begin{bmatrix} 
                        \cos{\frac{\theta}{2}} & -\sin{\frac{\theta}{2}} \\
                        \sin{\frac{\theta}{2}} & \cos{\frac{\theta}{2}} 
                      \end{bmatrix}

    :Effect:

    .. math::

        \begin{matrix}
            RY\left|0\right> = & \cos{\theta\over2}\left|0\right> -i\sin\frac{\theta}{2}\left|1\right> \\
            RY\left|1\right> =& -\sin\frac{\theta}{2}\left|0\right> + \cos\frac{\theta}{2}\left|1\right> 
        \end{matrix}

    :param q: Input qubits.
    :param theta: Rotation angle :math:`\theta`.
    """
    def __RY(q):
        q = reduce(add, q)
        _RY(theta, q)
        return q
      
    return __RY(q) if q is not None else __RY

def RZ(theta : float, q : Optional[quant | Iterable[quant]] = None) -> Callable[[quant], quant] | quant:
    r"""Z-axis rotation gate

    Apply a rotation of :math:`\theta` about the Z-axis on every qubit of ``q``.

    If ``q`` is not provided, return a new gate: ``RZ(theta)(q)``
    :math:`\equiv` ``RZ(theta, q)``. 

    :Matrix Representation:

    .. math::

         RZ(\theta) =  \begin{bmatrix} 
                            e^{-i\theta/2} & 0 \\
                            0 & e^{i\theta/2} 
                       \end{bmatrix}

    :Effect:

    .. math::

        \begin{matrix}
            RZ\left|0\right> = & e^{-i\theta/2}\left|0\right> \\ 
            RZ\left|1\right> =& e^{i\theta/2}\left|1\right> 
        \end{matrix}

    :param q: Input qubits.
    :param theta: Rotation angle :math:`\theta`.
    """
    def __RZ(q):
        q = reduce(add, q)
        _RZ(theta, q)
        return q
    
    return __RZ(q) if q is not None else __RZ


def I(q : quant | Iterable[quant]) -> quant:
    r"""Identity gate 

    Apply an identity gate on every qubit of ``q``.

    Note:
        This quantum gate does nothing! 

    :Matrix Representation:

    .. math::

        I = \begin{bmatrix} 
                1 & 0 \\
                0 & 1 
            \end{bmatrix}

    :Effect:

    .. math::

        \begin{matrix} 
            I\left|0\right> = & \left|0\right> \\
            I\left|1\right> = & \left|1\right>
        \end{matrix}

    :param q: Input qubits.
    """
    q = reduce(add, q)
    return q
 
def cnot(c : quant | Iterable[quant], t : quant | Iterable[quant]) -> tuple[quant, quant]:
    r"""Controlled-NOT
    
    Apply a CNOT gate between the qubits of c and t.

    :Matrix Representation:

    .. math::

        CNOT = \begin{bmatrix} 
                    1 & 0 & 0 & 0\\
                    0 & 1 & 0 & 0\\
                    0 & 0 & 0 & 1\\
                    0 & 0 & 1 & 0
                \end{bmatrix}

    :Effect:

    .. math::

        \begin{matrix} 
            CNOT\left|c t\right> = & \left|c (c\oplus t)\right> 
        \end{matrix}

    :param c: Control qubits.
    :param t: Target qubits.
    """
    
    c = reduce(add, c)
    t = reduce(add, t)
    
    for i, j in zip(c, t):
        ctrl(i, X, j)
    
    return c, t

def swap(a : quant | Iterable[quant], b : quant | Iterable[quant]) -> tuple[quant, quant]:
    r"""SWAP gate
    
    Swap the qubits of a and b.

    :Matrix Representation:

    .. math::

        CNOT = \begin{bmatrix} 
                    1 & 0 & 0 & 0\\
                    0 & 0 & 1 & 0\\
                    0 & 1 & 0 & 0\\
                    0 & 0 & 0 & 1
                \end{bmatrix}

    :Effect:

    .. math::

        \begin{matrix} 
            CNOT\left|a b\right> = & \left|b a\right> 
        \end{matrix}

    :param a: Input qubits.
    :param b: Input qubits.
    """
    a = reduce(add, a)
    b = reduce(add, b)

    cnot(a, b)
    cnot(b, a)
    cnot(a, b)

    return a, b

def RXX(theta : float, a : quant | Iterable[quant], b : quant | Iterable[quant]) -> tuple[quant, quant]:
    r"""XX-axis rotation gate
    

    :Matrix Representation:

    .. math::

        RXX(\theta) = \begin{bmatrix}
                          \cos\theta   & 0            & 0            & -i\sin\theta \\
                          0            & \cos\theta   & -i\sin\theta & 0 \\
                          0            & -i\sin\theta & \cos\theta   & 0 \\
                          -i\sin\theta & 0            & 0            & \cos\theta
                      \end{bmatrix}
  
    :param theta: :math:`\theta`.
    :param a: Input qubits.
    :param b: Input qubits.
    """
    a = reduce(add, a)
    b = reduce(add, b)
    
    for qa, qb in zip(a, b):
        with around([H, ctrl(0, X, 1)], qa+qb):
            RZ(theta, qb)

    return a, b 

def RYY(theta : float, a : quant | Iterable[quant], b : quant | Iterable[quant]) -> tuple[quant, quant]:
    r"""YY-axis rotation gate
    
    :param theta: :math:`\theta`.
    :param a: Input qubits.
    :param b: Input qubits.
    """
    a = reduce(add, a)
    b = reduce(add, b)
    
    for qa, qb in zip(a, b):
        with around([RX(pi/2), ctrl(0, X, 1)], qa+qb):
            RZ(theta, qb)

    return a, b 

def RZZ(theta : float, a : quant | Iterable[quant], b : quant | Iterable[quant]) -> tuple[quant, quant]:
    r"""ZZ-axis rotation gate
    
    :param theta: :math:`\theta`.
    :param a: Input qubits.
    :param b: Input qubits.
    """
    a = reduce(add, a)
    b = reduce(add, b)
    
    for qa, qb in zip(a, b):
        with around(ctrl(0, X, 1), qa+qb):
            RZ(theta, qb)
            
    return a, b 