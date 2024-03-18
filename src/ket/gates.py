"""Quantum gate definitions.

All quantum gates take one or more :class:`~ket.base.Quant` as input and return them at the end.
This allows for concatenating quantum operations.

Example:

    .. code-block:: python

        from ket import *

        p = Process()
        a, b = p.alloc(2)

        S(X(a))  # Apply a Pauli X followed by an S gate on `a`.

        CNOT(H(a), b)  # Apply a Hadamard on `a` followed by a CNOT gate on `a` and `b`.

For gates that take classical parameters, such as rotation gates, if non-qubits are passed,
it will return a new gate with the classical parameter set.

Example:

    .. code-block:: python

        from math import pi
        from ket import *

        s_gate = PHASE(pi/2)
        t_gate = PHASE(pi/4)

        p = Process()
        q = p.alloc()

        s_gate(q)
        t_gate(q)
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from fractions import Fraction
from functools import reduce
from math import pi
from operator import add
from typing import Callable

from .clib.libket import (
    HADAMARD,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    ROTATION_X,
    ROTATION_Y,
    ROTATION_Z,
    PHASE_SHIFT,
)

from .base import Quant
from .operations import ctrl, cat, kron, around

__all__ = [
    "I",
    "X",
    "Y",
    "Z",
    "H",
    "RX",
    "RY",
    "RZ",
    "PHASE",
    "S",
    "T",
    "SD",
    "TD",
    "CNOT",
    "SWAP",
    "RXX",
    "RZZ",
    "RYY",
]


def _gate_docstring(name, matrix, effect=None) -> str:
    return f"""Apply the {name} gate.
    
    .. csv-table::
        :delim: ;
        :header: Matrix{", Effect" if effect is not None else ""}

        :math:`{matrix}`{f"; :math:`{effect}`" if effect is not None else ""}
    """


def I(  # pylint: disable=invalid-name missing-function-docstring
    qubits: Quant,
) -> Quant:
    if not isinstance(qubits, Quant):
        qubits = reduce(add, qubits)

    return qubits


I.__doc__ = _gate_docstring(
    "Identity",
    r"\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}",
    r"\begin{matrix} I\left|0\right> = & \left|0\right> \\"
    r"I\left|1\right> = & \left|1\right> \end{matrix}",
)


def X(  # pylint: disable=invalid-name missing-function-docstring
    qubits: Quant,
) -> Quant:
    if not isinstance(qubits, Quant):
        qubits = reduce(add, qubits)

    for qubit in qubits.qubits:
        qubits.process.apply_gate(PAULI_X, 1, 1, 0.0, qubit)
    return qubits


X.__doc__ = _gate_docstring(
    "Pauli X",
    r"\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}",
    r"\begin{matrix} X\left|0\right> = & \left|1\right> \\"
    r"X\left|1\right> = & \left|0\right> \end{matrix}",
)


def Y(  # pylint: disable=invalid-name missing-function-docstring
    qubits: Quant,
) -> Quant:
    if not isinstance(qubits, Quant):
        qubits = reduce(add, qubits)

    for qubit in qubits.qubits:
        qubits.process.apply_gate(PAULI_Y, 1, 1, 0.0, qubit)
    return qubits


Y.__doc__ = _gate_docstring(
    "Pauli Y",
    r"\begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}",
    r"\begin{matrix} Y\left|0\right> = & i\left|1\right> \\"
    r"Y\left|1\right> = & -i\left|0\right> \end{matrix}",
)


def Z(  # pylint: disable=invalid-name missing-function-docstring
    qubits: Quant,
) -> Quant:
    if not isinstance(qubits, Quant):
        qubits = reduce(add, qubits)

    for qubit in qubits.qubits:
        qubits.process.apply_gate(PAULI_Z, 1, 1, 0.0, qubit)
    return qubits


Z.__doc__ = _gate_docstring(
    "Pauli Z",
    r"\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}",
    r"\begin{matrix} Z\left|0\right> = & \left|0\right> \\"
    r"Z\left|1\right> = & -\left|1\right> \end{matrix}",
)


def H(  # pylint: disable=invalid-name missing-function-docstring
    qubits: Quant,
) -> Quant:
    if not isinstance(qubits, Quant):
        qubits = reduce(add, qubits)

    for qubit in qubits.qubits:
        qubits.process.apply_gate(HADAMARD, 1, 1, 0.0, qubit)
    return qubits


H.__doc__ = _gate_docstring(
    "Hadamard",
    r"\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}",
    r"\begin{matrix} H\left|0\right> = & \frac{1}{\sqrt{2}}\left(\left|0\right> + \left|1\right>\right) \\"  # pylint: disable=line-too-long
    r"H\left|1\right> = & \frac{1}{\sqrt{2}}\left(\left|0\right> - \left|1\right>\right) \end{matrix}",  # pylint: disable=line-too-long
)


def RX(  # pylint: disable=invalid-name missing-function-docstring
    theta: float, qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:
    top, bottom = Fraction(theta / pi).limit_denominator().as_integer_ratio()
    use_fraction = abs(pi * top / bottom - theta) < 1e-14
    params = (top, bottom, 0.0) if use_fraction else (0, 0, theta)

    def inner(qubits: Quant) -> Quant:
        if not isinstance(qubits, Quant):
            qubits = reduce(add, qubits)

        for qubit in qubits.qubits:
            qubits.process.apply_gate(ROTATION_X, *params, qubit)
        return qubits

    if qubits is None:
        return inner
    return inner(qubits)


RX.__doc__ = _gate_docstring(
    "X-axes rotation",
    r"\begin{bmatrix} \cos(\theta/2) & -i\sin(\theta/2) \\ -i\sin(\theta/2) & \cos(\theta/2) \end{bmatrix}",  # pylint: disable=line-too-long
    r"\begin{matrix} R_x\left|0\right> = & \cos(\theta/2)\left|0\right> + i\sin(\theta/2)\left|1\right> \\"  # pylint: disable=line-too-long
    r"R_x\left|1\right> = & -i\sin(\theta/2)\left|0\right> + \cos(\theta/2)\left|1\right> \end{matrix}",  # pylint: disable=line-too-long
)


def RY(  # pylint: disable=invalid-name missing-function-docstring
    theta: float, qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:
    top, bottom = Fraction(theta / pi).limit_denominator().as_integer_ratio()
    use_fraction = abs(pi * top / bottom - theta) < 1e-14
    params = (top, bottom, 0.0) if use_fraction else (0, 0, theta)

    def inner(qubits: Quant) -> Quant:
        if not isinstance(qubits, Quant):
            qubits = reduce(add, qubits)

        for qubit in qubits.qubits:
            qubits.process.apply_gate(ROTATION_Y, *params, qubit)
        return qubits

    if qubits is None:
        return inner
    return inner(qubits)


RY.__doc__ = _gate_docstring(
    "Y-axes rotation",
    r"\begin{bmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{bmatrix}",  # pylint: disable=line-too-long
    r"\begin{matrix} R_y\left|0\right> = & \cos(\theta/2)\left|0\right> - \sin(\theta/2)\left|1\right> \\"  # pylint: disable=line-too-long
    r"R_y\left|1\right> = & \sin(\theta/2)\left|0\right> + \cos(\theta/2)\left|1\right> \end{matrix}",  # pylint: disable=line-too-long
)


def RZ(  # pylint: disable=invalid-name missing-function-docstring
    theta: float, qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:
    top, bottom = Fraction(theta / pi).limit_denominator().as_integer_ratio()
    use_fraction = abs(pi * top / bottom - theta) < 1e-14
    params = (top, bottom, 0.0) if use_fraction else (0, 0, theta)

    def inner(qubits: Quant) -> Quant:
        if not isinstance(qubits, Quant):
            qubits = reduce(add, qubits)

        for qubit in qubits.qubits:
            qubits.process.apply_gate(ROTATION_Z, *params, qubit)
        return qubits

    if qubits is None:
        return inner
    return inner(qubits)


RZ.__doc__ = _gate_docstring(
    "Z-axes rotation",
    r"\begin{bmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{bmatrix}",
    r"\begin{matrix} R_z\left|0\right> = & e^{-i\theta/2}\left|0\right> \\"
    r"R_z\left|1\right> = & e^{i\theta/2}\left|1\right> \end{matrix}",
)


def PHASE(  # pylint: disable=invalid-name missing-function-docstring
    theta: float, qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:
    top, bottom = Fraction(theta / pi).limit_denominator().as_integer_ratio()
    use_fraction = abs(pi * top / bottom - theta) < 1e-14
    params = (top, bottom, 0.0) if use_fraction else (0, 0, theta)

    def inner(qubits: Quant) -> Quant:
        if not isinstance(qubits, Quant):
            qubits = reduce(add, qubits)

        for qubit in qubits.qubits:
            qubits.process.apply_gate(PHASE_SHIFT, *params, qubit)
        return qubits

    if qubits is None:
        return inner
    return inner(qubits)


PHASE.__doc__ = _gate_docstring(
    "Phase shift",
    r"\begin{bmatrix} 1 & 0 \\ 0 & e^{i\theta} \end{bmatrix}",
    r"\begin{matrix} P\left|0\right> = & \left|0\right> \\"
    r"P\left|1\right> = & e^{i\theta}\left|1\right> \end{matrix}",
)


S = PHASE(pi / 2)
S.__doc__ = _gate_docstring(
    "S",
    r"\begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}",
    r"\begin{matrix} S\left|0\right> = & \left|0\right> \\"
    r"S\left|1\right> = & i\left|1\right> \end{matrix}",
)

SD = PHASE(-pi / 2)
SD.__doc__ = _gate_docstring(
    "S-dagger",
    r"\begin{bmatrix} 1 & 0 \\ 0 & -i \end{bmatrix}",
    r"\begin{matrix} S^\dagger\left|0\right> = & \left|0\right> \\"
    r"S^\dagger\left|1\right> = & -i\left|1\right> \end{matrix}",
)

T = PHASE(pi / 4)
T.__doc__ = _gate_docstring(
    "T",
    r"\begin{bmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{bmatrix}",
    r"\begin{matrix} T\left|0\right> = & \left|0\right> \\"
    r"T\left|1\right> = & e^{i\pi/4}\left|1\right> \end{matrix}",
)

TD = PHASE(-pi / 4)
TD.__doc__ = _gate_docstring(
    "T-dagger",
    r"\begin{bmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{bmatrix}",
    r"\begin{matrix} T^\dagger\left|0\right> = & \left|0\right> \\"
    r"T^\dagger\left|1\right> = & e^{-i\pi/4}\left|1\right> \end{matrix}",
)


def CNOT(  # pylint: disable=invalid-name missing-function-docstring
    control_qubit: Quant, target_qubit: Quant
) -> tuple[Quant, Quant]:
    for c, t in zip(control_qubit, target_qubit):
        ctrl(c, X)(t)
    return control_qubit, target_qubit


CNOT.__doc__ = _gate_docstring(
    "Controlled NOT",
    r"\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}",  # pylint: disable=line-too-long
    r"\begin{matrix} \text{CNOT}\left|00\right> = & \left|00\right> \\"
    r"\text{CNOT}\left|01\right> = & \left|01\right> \\"
    r"\text{CNOT}\left|10\right> = & \left|11\right> \\"
    r"\text{CNOT}\left|11\right> = & \left|10\right> \\"
    r"\text{CNOT}\left|\text{c}\right>\left|\text{t}\right> = & \left|\text{c}\right> \left|\text{c}\oplus\text{t}\right> \end{matrix}",  # pylint: disable=line-too-long
)


def SWAP(  # pylint: disable=invalid-name missing-function-docstring
    qubit_a: Quant, qubit_b: Quant
) -> tuple[Quant, Quant]:
    """Apply a SWAP gate."""
    return cat(CNOT, lambda a, b: (b, a), CNOT, lambda a, b: (b, a), CNOT)(
        qubit_a, qubit_b
    )


SWAP.__doc__ = _gate_docstring(
    "SWAP",
    r"\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}",  # pylint: disable=line-too-long
    r"\begin{matrix} \text{SWAP}\left|00\right> = & \left|00\right> \\"
    r"\text{SWAP}\left|01\right> = & \left|10\right> \\"
    r"\text{SWAP}\left|10\right> = & \left|01\right> \\"
    r"\text{SWAP}\left|11\right> = & \left|11\right> \\"
    r"\text{SWAP}\left|\text{a}\right>\left|\text{b}\right> = & \left|\text{b}\right> \left|\text{a}\right> \end{matrix}",  # pylint: disable=line-too-long
)


def RXX(  # pylint: disable=invalid-name missing-function-docstring
    theta: float, qubits_a: Quant | None, qubits_b: Quant | None
) -> tuple[Quant, Quant] | Callable[[Quant, Quant], tuple[Quant, Quant]]:
    def inner(qubits_a: Quant, qubits_b: Quant) -> tuple[Quant, Quant]:
        for qubit_a, qubit_b in zip(qubits_a, qubits_b):
            with around(cat(kron(H, H), CNOT), qubit_a, qubit_b):
                RZ(theta, qubit_b)

        return qubits_a, qubits_b

    if qubits_a is None and qubits_b is None:
        return inner
    return inner(qubits_a, qubits_b)


RXX.__doc__ = _gate_docstring(
    "XX rotation",
    r"\begin{bmatrix} \cos\frac{\theta}{2} & 0 & 0 & -i\sin\frac{\theta}{2} \\"
    r"0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\"
    r"0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\"
    r"-i\sin\frac{\theta}{2} & 0 & 0 & \cos\frac{\theta}{2} \end{bmatrix}",
)


def RZZ(  # pylint: disable=invalid-name missing-function-docstring
    theta: float, qubits_a: Quant | None, qubits_b: Quant | None
) -> tuple[Quant, Quant] | Callable[[Quant, Quant], tuple[Quant, Quant]]:
    def inner(qubits_a: Quant, qubits_b: Quant) -> tuple[Quant, Quant]:
        for qubit_a, qubit_b in zip(qubits_a, qubits_b):
            with around(CNOT, qubit_a, qubit_b):
                RZ(theta, qubit_b)

        return qubits_a, qubits_b

    if qubits_a is None and qubits_b is None:
        return inner
    return inner(qubits_a, qubits_b)


RZZ.__doc__ = _gate_docstring(
    "ZZ rotation",
    r"\begin{bmatrix} e^{-i \frac{\theta}{2}} & 0 & 0 & 0 \\"
    r"0 & e^{i \frac{\theta}{2}} & 0 & 0\\"
    r" 0 & 0 & e^{i \frac{\theta}{2}} & 0 \\"
    r"0 & 0 & 0 & e^{-i \frac{\theta}{2}} \end{bmatrix}",
)


def RYY(  # pylint: disable=invalid-name missing-function-docstring
    theta: float, qubits_a: Quant | None, qubits_b: Quant | None
) -> tuple[Quant, Quant] | Callable[[Quant, Quant], tuple[Quant, Quant]]:
    def inner(qubits_a: Quant, qubits_b: Quant) -> tuple[Quant, Quant]:
        for qubit_a, qubit_b in zip(qubits_a, qubits_b):
            with around(cat(kron(RX(pi / 2), RX(pi / 2)), CNOT), qubit_a, qubit_b):
                RZ(theta, qubit_b)
        return qubits_a, qubits_b

    if qubits_a is None and qubits_b is None:
        return inner
    return inner(qubits_a, qubits_b)


RYY.__doc__ = _gate_docstring(
    "YY rotation",
    r"\begin{bmatrix} \cos\frac{\theta}{2} & 0 & 0 & i\sin\frac{\theta}{2} \\"
    r"0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\"
    r"0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\"
    r"i\sin\frac{\theta}{2} & 0 & 0 & \cos\frac{\theta}{2} \end{bmatrix}",
)
