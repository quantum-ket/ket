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

        s_gate = P(pi/2)
        t_gate = P(pi/4)

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

# pylint: disable=protected-access

from contextlib import contextmanager
from math import isclose, pi, prod
from functools import reduce, wraps
from typing import Any, Callable
import functools
import contextvars

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

from .base import Process, Quant, Parameter
from .operations import (
    _search_process,
    control,
    ctrl,
    cat,
    kron,
    around,
    _allow_permutation,
    _unsafe_aux,
)

from .expv import Pauli, Hamiltonian

__all__ = [
    "I",
    "X",
    "Y",
    "Z",
    "B",
    "H",
    "RX",
    "RY",
    "RZ",
    "P",
    "S",
    "T",
    "SD",
    "TD",
    "U3",
    "CNOT",
    "CZ",
    "SWAP",
    "RXX",
    "RZZ",
    "RYY",
    "SX",
    "global_phase",
    "RBS",
    "obs",
    "QFT",
    "evolve",
]


def _gate_docstring(name, matrix, effect=None) -> str:
    return f"""Apply the {name} gate.
    
    .. csv-table::
        :delim: ;
        :header: Matrix{"; Effect" if effect is not None else ""}

        :math:`{matrix}`{f"; :math:`{effect}`" if effect is not None else ""}
    """


_build_obs = contextvars.ContextVar("build_obs", default=False)


@contextmanager
def obs():
    """
    Context manager to define a observable in Ket.

    When used within a ``with obs():`` block, any operator expressions
    constructed (e.g., sums of Pauli terms) are interpreted as part
    of a observable definition.

    This enables a more natural and symbolic style for building
    observable, closely mirroring their mathematical form.

    Example:

        .. code-block:: python

            with obs():
                h_c = -0.5 * sum(1 - Z(i) * Z(j) for i, j in edges)

    """
    token = _build_obs.set(True)
    try:
        yield
    finally:
        _build_obs.reset(token)


def I(  # pylint: disable=invalid-name missing-function-docstring
    qubits: Quant,
) -> Quant:
    if _build_obs.get():
        return Pauli.i(qubits)

    if not isinstance(qubits, Quant):
        qubits = reduce(Quant.__add__, qubits)

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
    if _build_obs.get():
        return Pauli.x(qubits)

    if not isinstance(qubits, Quant):
        qubits = reduce(Quant.__add__, qubits)

    process = qubits.ket_process

    is_diag = _is_diagonal.get()
    allow_perm = _allow_permutation.get()
    unsafe_aux = _unsafe_aux.get()

    aux_allowed = is_diag or allow_perm

    for qubit in qubits.qubits:
        is_aux = process._is_aux(qubit)

        if not unsafe_aux and (
            process._is_blocked(qubit) or (is_aux and not aux_allowed)
        ):
            raise RuntimeError("Gate application blocked for safe uncomputation")

        if is_aux and not is_diag:
            process._block_ctrl()

        process.apply_gate(PAULI_X, 0.0, False, 0, qubit)
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
    if _build_obs.get():
        return Pauli.y(qubits)

    if not isinstance(qubits, Quant):
        qubits = reduce(Quant.__add__, qubits)

    process = qubits.ket_process

    is_diag = _is_diagonal.get()
    allow_perm = _allow_permutation.get()
    unsafe_aux = _unsafe_aux.get()

    aux_allowed = is_diag or allow_perm

    for qubit in qubits.qubits:
        is_aux = process._is_aux(qubit)

        if not unsafe_aux and (
            process._is_blocked(qubit) or (is_aux and not aux_allowed)
        ):
            raise RuntimeError("Gate application blocked for safe uncomputation")

        if is_aux and not is_diag:
            process._block_ctrl()

        process.apply_gate(PAULI_Y, 0.0, False, 0, qubit)
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
    if _build_obs.get():
        return Pauli.z(qubits)

    if not isinstance(qubits, Quant):
        qubits = reduce(Quant.__add__, qubits)

    process = qubits.ket_process

    for qubit in qubits.qubits:
        process.apply_gate(PAULI_Z, 0.0, False, 0, qubit)
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
        qubits = reduce(Quant.__add__, qubits)

    process = qubits.ket_process

    is_diag = _is_diagonal.get()
    is_perm = _is_permutation.get()
    allow_perm = _allow_permutation.get()
    unsafe_aux = _unsafe_aux.get()

    aux_allowed = is_diag or (is_perm and allow_perm)

    for qubit in qubits.qubits:
        is_aux = process._is_aux(qubit)

        if not unsafe_aux and (
            process._is_blocked(qubit) or (is_aux and not aux_allowed)
        ):
            raise RuntimeError("Gate application blocked for safe uncomputation")

        if is_aux and not is_diag:
            process._block_ctrl()

        process.apply_gate(HADAMARD, 0.0, False, 0, qubit)
    return qubits


H.__doc__ = _gate_docstring(
    "Hadamard",
    r"\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}",
    r"\begin{matrix}"
    r"H\left|0\right> = & \frac{1}{\sqrt{2}}\left(\left|0\right> + \left|1\right>\right) \\"
    r"H\left|1\right> = & \frac{1}{\sqrt{2}}\left(\left|0\right> - \left|1\right>\right)"
    r"\end{matrix}",
)


def _isclose_mod(value: float, target: float, tolerance: float = 1e-8) -> bool:
    remainder = value % target

    is_zero = isclose(remainder, 0.0, abs_tol=tolerance)
    is_target = isclose(remainder, target, abs_tol=tolerance)

    return is_zero or is_target


def RX(  # pylint: disable=invalid-name missing-function-docstring
    theta: float | Parameter, qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:

    if not isinstance(theta, Parameter):
        theta_diagonal = _isclose_mod(theta, 2 * pi)
        theta_permutation = _isclose_mod(theta, pi)
    else:
        theta_diagonal = False
        theta_permutation = False

    def inner(qubits: Quant) -> Quant:
        if not isinstance(qubits, Quant):
            qubits = reduce(Quant.__add__, qubits)

        process = qubits.ket_process

        is_diag = _is_diagonal.get() or theta_diagonal
        is_perm = _is_permutation.get() or theta_permutation
        allow_perm = _allow_permutation.get()
        unsafe_aux = _unsafe_aux.get()

        aux_allowed = is_diag or (is_perm and allow_perm)

        for qubit in qubits.qubits:
            is_aux = process._is_aux(qubit)

            if not unsafe_aux and (
                process._is_blocked(qubit) or (is_aux and not aux_allowed)
            ):
                raise RuntimeError("Gate application blocked for safe uncomputation")

            if is_aux and not is_diag:
                process._block_ctrl()

            if isinstance(theta, Parameter):
                process.apply_gate(
                    ROTATION_X,
                    theta._multiplier,
                    True,
                    theta._index,
                    qubit,
                )
            else:
                process.apply_gate(
                    ROTATION_X,
                    theta,
                    False,
                    0,
                    qubit,
                )
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

    if not isinstance(theta, Parameter):
        theta_diagonal = _isclose_mod(theta, 2 * pi)
        theta_permutation = _isclose_mod(theta, pi)
    else:
        theta_diagonal = False
        theta_permutation = False

    def inner(qubits: Quant) -> Quant:
        if not isinstance(qubits, Quant):
            qubits = reduce(Quant.__add__, qubits)

        process = qubits.ket_process

        is_diag = _is_diagonal.get() or theta_diagonal
        is_perm = _is_permutation.get() or theta_permutation
        allow_perm = _allow_permutation.get()
        unsafe_aux = _unsafe_aux.get()

        aux_allowed = is_diag or (is_perm and allow_perm)

        for qubit in qubits.qubits:
            is_aux = process._is_aux(qubit)

            if not unsafe_aux and (
                process._is_blocked(qubit) or (is_aux and not aux_allowed)
            ):
                raise RuntimeError("Gate application blocked for safe uncomputation")

            if is_aux and not is_diag:
                process._block_ctrl()

            if isinstance(theta, Parameter):
                process.apply_gate(
                    ROTATION_Y,
                    theta._multiplier,
                    True,
                    theta._index,
                    qubit,
                )
            else:
                process.apply_gate(
                    ROTATION_Y,
                    theta,
                    False,
                    0,
                    qubit,
                )
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
            qubits = reduce(Quant.__add__, qubits)

        process = qubits.ket_process

        for qubit in qubits.qubits:
            if isinstance(theta, Parameter):
                process.apply_gate(
                    ROTATION_Z,
                    theta._multiplier,
                    True,
                    theta._index,
                    qubit,
                )
            else:
                process.apply_gate(
                    ROTATION_Z,
                    theta,
                    False,
                    0,
                    qubit,
                )
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


def P(  # pylint: disable=invalid-name missing-function-docstring
    theta: float, qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:

    def inner(qubits: Quant) -> Quant:
        if not isinstance(qubits, Quant):
            qubits = reduce(Quant.__add__, qubits)

        process = qubits.ket_process

        for qubit in qubits.qubits:
            if isinstance(theta, Parameter):
                process.apply_gate(
                    PHASE_SHIFT,
                    theta._multiplier,
                    True,
                    theta._index,
                    qubit,
                )
            else:
                process.apply_gate(
                    PHASE_SHIFT,
                    theta,
                    False,
                    0,
                    qubit,
                )
        return qubits

    if qubits is None:
        return inner
    return inner(qubits)


P.__doc__ = _gate_docstring(
    "Phase shift",
    r"\begin{bmatrix} 1 & 0 \\ 0 & e^{i\theta} \end{bmatrix}",
    r"\begin{matrix} P\left|0\right> = & \left|0\right> \\"
    r"P\left|1\right> = & e^{i\theta}\left|1\right> \end{matrix}",
)


S = P(pi / 2)
S.__doc__ = _gate_docstring(
    "S",
    r"\begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}",
    r"\begin{matrix} S\left|0\right> = & \left|0\right> \\"
    r"S\left|1\right> = & i\left|1\right> \end{matrix}",
)

SD = P(-pi / 2)
SD.__doc__ = _gate_docstring(
    "S-dagger",
    r"\begin{bmatrix} 1 & 0 \\ 0 & -i \end{bmatrix}",
    r"\begin{matrix} S^\dagger\left|0\right> = & \left|0\right> \\"
    r"S^\dagger\left|1\right> = & -i\left|1\right> \end{matrix}",
)

T = P(pi / 4)
T.__doc__ = _gate_docstring(
    "T",
    r"\begin{bmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{bmatrix}",
    r"\begin{matrix} T\left|0\right> = & \left|0\right> \\"
    r"T\left|1\right> = & e^{i\pi/4}\left|1\right> \end{matrix}",
)

TD = P(-pi / 4)
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


def CZ(  # pylint: disable=invalid-name missing-function-docstring
    *qubits: list[Quant],
) -> list[Quant, Quant]:
    for q in zip(*qubits):
        ctrl(q[:-1], Z)(q[-1])
    return qubits


CZ.__doc__ = _gate_docstring(
    "Multi-Controlled Z",
    r"\begin{bmatrix}"
    r"1 & 0 & \cdots & 0 & 0 \\ 0 & 1 & \cdots & 0 & 0 \\"
    r"\vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & \cdots & 1 & 0 \\"
    r"0 & 0 & \cdots & 0 & -1"
    r"\end{bmatrix}",
    r"\begin{matrix}"
    r"\text{CZ}\left|0\cdots0\right> = & \left|0\cdots0\right> \\"
    r"\text{CZ}\left|0\cdots1\right> = & \left|0\cdots1\right> \\"
    r"\text{CZ}\left|1\cdots0\right> = & \left|1\cdots0\right> \\"
    r"\text{CZ}\left|1\cdots1\right> = & -\left|1\cdots1\right> \\"
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
    theta: float, qubits_a: Quant | None = None, qubits_b: Quant | None = None
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
    theta: float, qubits_a: Quant | None = None, qubits_b: Quant | None = None
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
    theta: float, qubits_a: Quant | None = None, qubits_b: Quant | None = None
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
    "RYY rotation",
    r"\begin{bmatrix}"
    r"\cos\frac{\theta}{2} & 0 & 0 & i\sin\frac{\theta}{2} \\"
    r"0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\"
    r"0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\"
    r"i\sin\frac{\theta}{2} & 0 & 0 & \cos\frac{\theta}{2}"
    r"\end{bmatrix}",
)


def RBS(  # pylint: disable=invalid-name missing-function-docstring
    theta: float, qubits_a: Quant | None = None, qubits_b: Quant | None = None
) -> tuple[Quant, Quant] | Callable[[Quant, Quant], tuple[Quant, Quant]]:
    def inner(qubits_a: Quant, qubits_b: Quant) -> tuple[Quant, Quant]:
        for qubit_a, qubit_b in zip(qubits_a, qubits_b):
            with around(cat(kron(H, H), CNOT), qubit_a, qubit_b):
                RY(theta / 2, qubit_a)
                RY(-theta / 2, qubit_b)
        return qubits_a, qubits_b

    if qubits_a is None and qubits_b is None:
        return inner
    return inner(qubits_a, qubits_b)


RBS.__doc__ = _gate_docstring(
    "Reconfigurable Beam Splitter (RBS)",
    r"\begin{bmatrix}"
    r"1 & 0 & 0 & 0 \\"
    r"0 & \cos\theta & \sin\theta & 0 \\"
    r"0 & -\sin\theta & \cos\theta & 0 \\"
    r"0 & 0 & 0 & 1"
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
        @functools.wraps(gate)
        def inner(*args, ket_process: Process | None = None, **kwargs):
            ket_process = _search_process(ket_process, args, kwargs)

            ket_process.apply_global_phase(theta)

            return gate(*args, **kwargs)

        return inner

    return _global_phase


SX = global_phase(pi / 4)(RX(pi / 2))  # pylint: disable=invalid-name
SX.__doc__ = _gate_docstring(
    "Sqrt X",
    r"\frac{1}{2} \begin{bmatrix} 1+i & 1-i \\ 1-i & 1+i \end{bmatrix}",
    r"\begin{matrix}"
    r"\sqrt{X}\left|0\right> = & \frac{1}{2} ((1+i)\left|0\right> + (1-i)\left|1\right>) \\"
    r"\sqrt{X}\left|1\right> = & \frac{1}{2} ((1-i)\left|0\right> + (1+i)\left|1\right>)"
    r"\end{matrix}",
)


def QFT(qubits, do_swap: bool = True):  # pylint: disable=invalid-name
    r"""Quantum Fourier Transform.

    .. math::

        \text{QFT}\left|x\right> =
        \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{2\pi i x k / N} \left|k\right>

    args:
        qubits: The qubits to apply the QFT to.
        do_swap: Whether to invert the qubits after the transformation.
    """
    if len(qubits) == 1:
        H(qubits)
    else:
        *init, last = qubits
        H(last)

        for i, ctrl_qubit in enumerate(reversed(init)):
            with control(ctrl_qubit):
                P(pi / 2 ** (i + 1), last)

        QFT(init, do_swap=False)

    if do_swap:
        size = len(qubits)
        for i in range(size // 2):
            SWAP(qubits[i], qubits[size - i - 1])


def B(qubit):  # pylint: disable=invalid-name
    r"""Binary-encoded observable.

    Construct an observable that has eigenvalue 0 for :math:`\left|0\right>`
    and 1 for :math:`\left|1\right>`. This is useful for QUBO
    Hamiltonian construction.

    .. math::

        \frac{1 - Z}{2}

    """

    with obs():
        return prod((1 - Z(q)) / 2 for q in qubit)


_ROT_MAP = {
    "X": H,
    "Y": RX(pi / 2),
    "Z": I,
}


def _rot(string, qubits):
    for p, q in zip(string, qubits):
        _ROT_MAP[p](q)


def _rzg(angle, qubits):
    if len(qubits) == 1:
        RZ(angle, qubits)
    else:
        with around(CNOT, qubits[0], qubits[1]):
            _rzg(angle, qubits[1:])


_RGATE = {
    "X": RX,
    "Y": RY,
    "Z": RZ,
}


def evolve(hamiltonian: Hamiltonian):
    """Evolve the quantum state according to the given Hamiltonian."""
    process = hamiltonian.ket_process
    for term in hamiltonian.terms:
        gates_qubits = [
            (gate, qubit) for qubit, gate in term.map.items() if gate != "I"
        ]

        if not gates_qubits:
            continue

        gates, qubits = zip(*gates_qubits)

        qubits = Quant(qubits=list(qubits), process=process)

        if len(gates) == 1:
            _RGATE[gates[0]](2 * term.coef, qubits)
        else:
            with around(_rot, gates, qubits):
                _rzg(2 * term.coef, qubits)


_is_diagonal = contextvars.ContextVar("is_diagonal", default=False)


def is_diagonal(gate: Callable) -> Callable:
    """Force to consider as a diagonal gate.

    Args:
        gate: Quantum gate
    """

    @wraps(gate)
    def inner(*args, **kwargs) -> Any:
        token = _is_diagonal.set(True)
        try:
            return gate(*args, **kwargs)
        finally:
            _is_diagonal.reset(token)

    return inner


_is_permutation = contextvars.ContextVar("is_permutation", default=False)


def is_permutation(gate: Callable) -> Callable:
    """Force to consider as a permutation gate.

    Args:
        gate: Quantum gate
    """

    @wraps(gate)
    def inner(*args, **kwargs) -> Any:
        token = _is_permutation.set(True)
        try:
            return gate(*args, **kwargs)
        finally:
            _is_permutation.reset(token)

    return inner
