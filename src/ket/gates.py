"""Quantum gate definitions.

All quantum gates accept one or more :class:`~ket.base.Quant` objects as input
and return them unchanged, enabling method chaining:

.. code-block:: python

    from ket import *

    p = Process()
    a, b = p.alloc(2)

    S(X(a))           # Apply X then S on qubit `a`.
    CNOT(H(a), b)     # Apply H on `a`, then CNOT with `a` as control.

For parameterized gates (e.g., rotation gates), **partial application** is
supported: if the qubit argument is omitted, a gate callable is returned
instead of being applied immediately:

.. code-block:: python

    from math import pi
    from ket import *

    # Create reusable gate instances with pre-set angles
    s_gate = P(pi / 2)   # returns a callable, not applied yet
    t_gate = P(pi / 4)

    p = Process()
    q = p.alloc()

    s_gate(q)             # now applied
    t_gate(q)

Gates that accept two-qubit arguments (e.g., :func:`~ket.gates.RXX`,
:func:`~ket.gates.CNOT`) also support partial application: passing only the
angle returns a two-qubit gate callable.

**Observable mode** (``with obs():``):
Several single-qubit gate functions (:func:`~ket.gates.X`, :func:`~ket.gates.Y`,
:func:`~ket.gates.Z`, :func:`~ket.gates.I`) have a dual role: inside an
:func:`~ket.gates.obs` context manager block they construct
:class:`~ket.expv.Pauli` objects for Hamiltonian building rather than
applying the physical gate to the circuit.
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

from ket.clib.libket import search_process

from .base import Process, Quant, Parameter
from .operations import (
    control,
    ctrl,
    cat,
    kron,
    around,
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
    r"""Context manager for constructing quantum observables in symbolic form.

    Inside a ``with obs():`` block, the Pauli gate functions
    (:func:`~ket.gates.X`, :func:`~ket.gates.Y`, :func:`~ket.gates.Z`,
    :func:`~ket.gates.I`) return :class:`~ket.expv.Pauli` objects instead of
    applying the physical gate to the circuit. These objects can be combined
    with arithmetic operators to build :class:`~ket.expv.Hamiltonian` objects
    suitable for :func:`~ket.operations.exp_value`.

    This approach mirrors the mathematical notation for observables and
    avoids manually constructing :class:`~ket.expv.Pauli` instances.

    Example:

        .. code-block:: python

            from ket import *

            p = Process()
            q = p.alloc(4)

            edges = [(q[0], q[1]), (q[1], q[2]), (q[2], q[3])]

            with obs():
                # MaxCut QUBO cost Hamiltonian
                h_c = -0.5 * sum(1 - Z(i) * Z(j) for i, j in edges)

            ev = exp_value(h_c)
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

    with process.block_builder() as block:
        for qubit in qubits.qubits:
            block.append_gate("PauliX", qubit)
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

    with process.block_builder() as block:
        for qubit in qubits.qubits:
            block.append_gate("PauliY", qubit)
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

    with process.block_builder() as block:
        for qubit in qubits.qubits:
            block.append_gate("PauliZ", qubit)

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

    with process.block_builder() as block:
        for qubit in qubits.qubits:
            block.append_gate("Hadamard", qubit)

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

    def inner(qubits: Quant) -> Quant:
        if not isinstance(qubits, Quant):
            qubits = reduce(Quant.__add__, qubits)

        process = qubits.ket_process

        with process.block_builder() as block:
            for qubit in qubits.qubits:
                if isinstance(theta, Parameter):
                    gate = {
                        "RotationX": {
                            "Ref": {
                                "index": theta._index,
                                "multiplier": theta._multiplier,
                                "value": theta._param,
                            }
                        }
                    }
                else:
                    gate = {"RotationX": {"Value": theta}}
                block.append_gate(gate, qubit)

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
    theta: float | Parameter, qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:

    def inner(qubits: Quant) -> Quant:
        if not isinstance(qubits, Quant):
            qubits = reduce(Quant.__add__, qubits)

        process = qubits.ket_process

        with process.block_builder() as block:
            for qubit in qubits.qubits:
                if isinstance(theta, Parameter):
                    gate = {
                        "RotationY": {
                            "Ref": {
                                "index": theta._index,
                                "multiplier": theta._multiplier,
                                "value": theta._param,
                            }
                        }
                    }
                else:
                    gate = {"RotationY": {"Value": theta}}
                block.append_gate(gate, qubit)

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
    theta: float | Parameter, qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:

    def inner(qubits: Quant) -> Quant:
        if not isinstance(qubits, Quant):
            qubits = reduce(Quant.__add__, qubits)

        process = qubits.ket_process

        with process.block_builder() as block:
            for qubit in qubits.qubits:
                if isinstance(theta, Parameter):
                    gate = {
                        "RotationZ": {
                            "Ref": {
                                "index": theta._index,
                                "multiplier": theta._multiplier,
                                "value": theta._param,
                            }
                        }
                    }
                else:
                    gate = {"RotationZ": {"Value": theta}}
                block.append_gate(gate, qubit)

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
    theta: float | Parameter, qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:

    def inner(qubits: Quant) -> Quant:
        if not isinstance(qubits, Quant):
            qubits = reduce(Quant.__add__, qubits)

        process = qubits.ket_process

        with process.block_builder() as block:
            for qubit in qubits.qubits:
                if isinstance(theta, Parameter):
                    gate = {
                        "Phase": {
                            "Ref": {
                                "index": theta._index,
                                "multiplier": theta._multiplier,
                                "value": theta._param,
                            }
                        }
                    }
                else:
                    gate = {"Phase": {"Value": theta}}
                block.append_gate(gate, qubit)

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
    *qubits: Quant,
) -> tuple[Quant, ...]:
    if not qubits:
        return qubits

    all_qubits = reduce(Quant.__add__, qubits)

    if len(all_qubits) > 1:
        ctrl(all_qubits[:-1], Z)(all_qubits[-1])
    elif len(all_qubits) == 1:
        Z(all_qubits)

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
            with around(cat(kron(H, I), CNOT), qubit_a, qubit_b):
                RY(theta / 2, qubit_a)
                RY(theta / 2, qubit_b)
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
            ket_process = search_process(ket_process, args, kwargs)
            with ket_process.block_builder() as block:
                ret = gate(*args, **kwargs)
                block.add_global_phase(theta)
            return ret

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
    r"""Apply the Quantum Fourier Transform (QFT) to the given qubits.

    Implements the standard recursive QFT circuit:

    .. math::

        \text{QFT}\left|x\right\rangle =
        \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{2\pi i x k / N} \left|k\right\rangle

    where :math:`N = 2^n` and :math:`n` is the number of qubits.

    The circuit consists of Hadamard gates and controlled phase rotations,
    followed by an optional SWAP network that puts the output in the standard
    bit-order (most-significant qubit first). Set ``do_swap=False`` when the
    SWAP is handled externally.

    Example:

        .. code-block:: python

            from ket import *

            p = Process(simulator="dense", num_qubits=4)
            q = p.alloc(4)

            # Initialize to |1⟩ and apply QFT
            X(q[3])
            QFT(q)

            state = dump(q)
            print(state.show())

    Args:
        qubits: The register to apply the QFT to.
            Must have at least 1 qubit.
        do_swap: If ``True`` (default), applies the qubit-reversal SWAP
            network at the end so the output is in standard bit order.
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


def B(qubit: Quant):  # pylint: disable=invalid-name
    r"""Construct a binary-encoded QUBO observable for a qubit register.

    Returns a :class:`~ket.expv.Hamiltonian` with eigenvalue ``0`` for
    :math:`\left|0\right\rangle` and eigenvalue ``1`` for
    :math:`\left|1\right\rangle`. When applied to a register, the result
    is the tensor product, giving eigenvalue equal to the binary integer
    encoded in the register:

    .. math::

        B = \frac{\mathbf{1} - Z}{2}

    This observable is essential for building **QUBO Hamiltonians** where the
    cost function is a polynomial over binary variables.

    Example:

        .. code-block:: python

            from ket import *

            p = Process()
            q = p.alloc(3)

            # QUBO cost: x0 + x1*x2 - 2*x0*x2
            cost = B(q[0]) + B(q[1]) * B(q[2]) - 2 * B(q[0]) * B(q[2])

    Args:
        qubit: The qubit(s) to apply the observable
            to. If a multi-qubit register is passed, the result is the tensor
            product of :math:`B` over all qubits.

    Returns:
        The binary-encoded observable.
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


_RGATE = {"X": RX, "Y": RY, "Z": RZ}


def evolve(hamiltonian: Hamiltonian):
    r"""Time evolution :math:`e^{-iHt}` for a Hamiltonian.

    For each Pauli term :math:`c_k P_k` in the Hamiltonian, applies the
    corresponding single-step Trotter evolution gate
    :math:`e^{-i c_k P_k}` to the qubits in that term.

    .. note::
        This implements a single Trotter step, **not** a full time evolution.
        For accurate dynamics, you may need to call ``evolve`` multiple times
        with a small coefficient or compose it with other techniques.

    Example:

        .. code-block:: python

            from ket import *

            p = Process()
            q = p.alloc(2)

            with obs():
                # Transverse-field Ising: -J Z0Z1 - h X0
                H_ising = -1.0 * Z(q[0]) * Z(q[1]) - 0.5 * X(q[0])

            evolve(H_ising)   # one Trotter step with t encoded in the coefficients

    Args:
        hamiltonian: The Hamiltonian to
            simulate. The coefficient of each term serves as the rotation
            angle.
    """
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


def is_diagonal(gate: Callable) -> Callable:
    """Decorator that marks a gate as diagonal in the computational basis.

    This tells the Ket runtime that the gate only adds phases to basis states
    without permuting them, which relaxes certain uncomputation safety checks.

    Use this decorator when you have implemented a custom gate that is
    provably diagonal (e.g., a phase oracle or a controlled-phase network)
    and you want the runtime to recognize and exploit that property.

    Example:

        .. code-block:: python

            from ket import *

            @is_diagonal
            def phase_oracle(qubits):
                # Flips the phase of the |111> state.
                ctrl(qubits[:-1], Z)(qubits[-1])

    Args:
        gate: The quantum gate function to decorate.

    Returns:
        A wrapped version of ``gate`` whose block is flagged as diagonal.
    """

    @wraps(gate)
    def inner(*args, **kwargs) -> Any:
        process = search_process(None, args, kwargs)
        with process.block_builder(diagonal=True):
            return gate(*args, **kwargs)

    return inner


def is_permutation(gate: Callable) -> Callable:
    """Decorator that marks a gate as a permutation of computational basis states.

    This tells the Ket runtime that the gate maps
    each basis state to exactly one other basis state (i.e., it is a
    classical reversible function), which enables optimizations and loosens
    uncomputation restrictions on auxiliary qubits written by permutations.

    Use this decorator for gates that implement reversible classical logic
    (e.g., arithmetic adders, LUT-based oracles, swap networks).

    Example:

        .. code-block:: python

            from ket import *

            @is_permutation
            def increment(qubits):
                # Increment a binary register modulo 2^n.
                for i, q in enumerate(qubits):
                    ctrl(qubits[:i], X)(q)

    Args:
        gate: The quantum gate function to decorate.

    Returns:
        A wrapped version of ``gate`` whose block is flagged as a permutation.
    """

    @wraps(gate)
    def inner(*args, **kwargs) -> Any:
        process = search_process(None, args, kwargs)
        with process.block_builder(permutation=True):
            return gate(*args, **kwargs)

    return inner
