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

from math import pi
from functools import reduce
from operator import add
from typing import Any, Callable

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

from .base import Process, Quant
from .operations import _search_process, ctrl, cat, kron, around

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
    "U3",
    "CNOT",
    "SWAP",
    "RXX",
    "RZZ",
    "RYY",
    "SX",
    "global_phase",
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
        qubits.process.apply_gate(PAULI_X, 0.0, qubit)
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
        qubits.process.apply_gate(PAULI_Y, 0.0, qubit)
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
        qubits.process.apply_gate(PAULI_Z, 0.0, qubit)
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
        qubits.process.apply_gate(HADAMARD, 0.0, qubit)
    return qubits


H.__doc__ = _gate_docstring(
    "Hadamard",
    r"\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}",
    r"\begin{matrix}"
    r"H\left|0\right> = & \frac{1}{\sqrt{2}}\left(\left|0\right> + \left|1\right>\right) \\"
    r"H\left|1\right> = & \frac{1}{\sqrt{2}}\left(\left|0\right> - \left|1\right>\right)"
    r"\end{matrix}",
)


def RX(  # pylint: disable=invalid-name missing-function-docstring
    theta: float, qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:

    def inner(qubits: Quant) -> Quant:
        if not isinstance(qubits, Quant):
            qubits = reduce(add, qubits)

        for qubit in qubits.qubits:
            qubits.process.apply_gate(ROTATION_X, theta, qubit)
        return qubits

    if qubits is None:
        return inner
    return inner(qubits)


RX.__doc__ = _gate_docstring(
    "X-axes rotation",
    r"\begin{bmatrix}"
    r"\cos(\theta/2) & -i\sin(\theta/2) \\ -i\sin(\theta/2) & \cos(\theta/2)"
    r"\end{bmatrix}",
    r"\begin{matrix}"
    r"R_x\left|0\right> = & \cos(\theta/2)\left|0\right> - i\sin(\theta/2)\left|1\right> \\"
    r"R_x\left|1\right> = & -i\sin(\theta/2)\left|0\right> + \cos(\theta/2)\left|1\right>"
    r"\end{matrix}",
)


def RY(  # pylint: disable=invalid-name missing-function-docstring
    theta: float, qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:

    def inner(qubits: Quant) -> Quant:
        if not isinstance(qubits, Quant):
            qubits = reduce(add, qubits)

        for qubit in qubits.qubits:
            qubits.process.apply_gate(ROTATION_Y, theta, qubit)
        return qubits

    if qubits is None:
        return inner
    return inner(qubits)


RY.__doc__ = _gate_docstring(
    "Y-axes rotation",
    r"\begin{bmatrix}"
    r"\cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2)"
    r"\end{bmatrix}",
    r"\begin{matrix}"
    r"R_y\left|0\right> = & \cos(\theta/2)\left|0\right> + \sin(\theta/2)\left|1\right> \\"
    r"R_y\left|1\right> = & -\sin(\theta/2)\left|0\right> + \cos(\theta/2)\left|1\right>"
    r"\end{matrix}",
)


def RZ(  # pylint: disable=invalid-name missing-function-docstring
    theta: float, qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:

    def inner(qubits: Quant) -> Quant:
        if not isinstance(qubits, Quant):
            qubits = reduce(add, qubits)

        for qubit in qubits.qubits:
            qubits.process.apply_gate(ROTATION_Z, theta, qubit)
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

    def inner(qubits: Quant) -> Quant:
        if not isinstance(qubits, Quant):
            qubits = reduce(add, qubits)

        for qubit in qubits.qubits:
            qubits.process.apply_gate(PHASE_SHIFT, theta, qubit)
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
    r"\begin{bmatrix}"
    r"1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0"
    r"\end{bmatrix}",
    r"\begin{matrix}"
    r"\text{CNOT}\left|00\right> = & \left|00\right> \\"
    r"\text{CNOT}\left|01\right> = & \left|01\right> \\"
    r"\text{CNOT}\left|10\right> = & \left|11\right> \\"
    r"\text{CNOT}\left|11\right> = & \left|10\right> \\"
    r"\text{CNOT}\left|\text{c}\right>\left|\text{t}\right> ="
    r"& \left|\text{c}\right> \left|\text{c}\oplus\text{t}\right>"
    r"\end{matrix}",
)


def SWAP(  # pylint: disable=invalid-name missing-function-docstring
    qubit_a: Quant, qubit_b: Quant
) -> tuple[Quant, Quant]:
    """Apply a SWAP gate."""
    with around(CNOT, qubit_a, qubit_b):
        CNOT(qubit_b, qubit_a)

    return qubit_a, qubit_b


SWAP.__doc__ = _gate_docstring(
    "SWAP",
    r"\begin{bmatrix}"
    r"1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1"
    r"\end{bmatrix}",
    r"\begin{matrix}"
    r"\text{SWAP}\left|00\right> = & \left|00\right> \\"
    r"\text{SWAP}\left|01\right> = & \left|10\right> \\"
    r"\text{SWAP}\left|10\right> = & \left|01\right> \\"
    r"\text{SWAP}\left|11\right> = & \left|11\right> \\"
    r"\text{SWAP}\left|\text{a}\right>\left|\text{b}\right> ="
    r"& \left|\text{b}\right> \left|\text{a}\right>"
    r"\end{matrix}",
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
    "ZZ rotation",
    r"\begin{bmatrix}"
    r"\cos\frac{\theta}{2} & 0 & 0 & i\sin\frac{\theta}{2} \\"
    r"0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\"
    r"0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\"
    r"i\sin\frac{\theta}{2} & 0 & 0 & \cos\frac{\theta}{2}"
    r"\end{bmatrix}",
)


def U3(  # pylint: disable=invalid-name missing-function-docstring
    theta: float, phi: float, lambda_: float, qubit: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:
    gate = cat(RZ(lambda_), RY(theta), RZ(phi))
    if qubit is not None:
        return gate(qubit)
    return gate


U3.__doc__ = _gate_docstring(
    "U3",
    r"\begin{bmatrix}"
    r"e^{-i (\phi + \lambda)/2} \cos(\theta/2) & -e^{-i (\phi - \lambda)/2} \sin(\theta/2) \\"
    r"e^{i (\phi - \lambda)/2} \sin(\theta/2) & e^{i (\phi + \lambda)/2} \cos(\theta/2)"
    r"\end{bmatrix}",
    r"\begin{matrix}"
    r"U3\left|0\right> = & e^{-i (\phi + \lambda)/2} \cos(\theta/2)\left|0\right>"
    r"+ e^{i (\phi - \lambda)/2} \sin(\theta/2) \left|1\right> \\"
    r"U3\left|1\right> = & -e^{-i (\phi - \lambda)/2} \sin(\theta/2)\left|0\right>"
    r"+ e^{i (\phi + \lambda)/2} \cos(\theta/2)\left|1\right> \\"
    r"\end{matrix}",
)


def global_phase(
    theta: float,
) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
    r"""Apply a global phase to a quantum operation.

    Decorator that adds a global phase :math:`e^{i\theta}` to a quantum gate
    :math:`U`, creating the gate :math:`e^{i\theta}U`.

    In quantum computation, global phases are overall factors that can be
    applied to quantum states without affecting the measurement outcomes.
    Mathematically, they represent rotations in the complex plane and are
    usually ignored because they have no observable consequences. However, in
    certain contexts, such as controlled quantum operations, the global phase
    can affect the behavior of the operation.

    The addition of a global phase can be important for maintaining
    consistency in quantum algorithms, particularly when dealing with controlled
    operations where relative phase differences between different components
    of the quantum state can impact the computation.

    Example:

    .. code-block:: python

        @global_phase(pi / 2)
        def my_z_gate(qubit):
            return RZ(pi, qubit)

    This example defines a custom quantum gate equivalent to a Pauli Z
    operation, where :math:`Z = e^{i\frac{\pi}{2}}R_z(\pi)`.

    Args:
        theta: The :math:`\theta` angle of the global phase :math:`e^{i\theta}`.
    """

    def _global_phase(gate: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def inner(*args, ket_process: Process | None = None, **kwargs):
            ket_process = _search_process(ket_process, args, kwargs)

            ket_process.apply_global_phase(theta)

            return gate(*args, **kwargs)

        return inner

    return _global_phase


SX = global_phase(pi / 4)(RX(pi / 2))
SX.__doc__ = _gate_docstring(
    "Sqrt X",
    r"\frac{1}{2} \begin{bmatrix} 1+i & 1-i \\ 1-i & 1+i \end{bmatrix}",
    r"\begin{matrix}"
    r"\sqrt{X}\left|0\right> = & \frac{1}{2} ((1+i)\left|0\right> + (1-i)\left|1\right>) \\"
    r"\sqrt{X}\left|1\right> = & \frac{1}{2} ((1-i)\left|0\right> + (1+i)\left|1\right>)"
    r"\end{matrix}",
)
