"""Quantum library.

Utilities for preparing quantum states and building quantum algorithms.
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from functools import reduce
from operator import add
from typing import Callable, Literal
from cmath import asin, exp, isclose, cos, sin
from math import acos, sqrt, atan2
from collections.abc import Sized

from ..base import Quant, Process
from ..operations import ctrl, around, dump
from ..gates import RZ, X, Z, H, RY, CNOT, S, global_phase

__all__ = [
    "flip_to_control",
    "phase_oracle",
    "prepare_ghz",
    "prepare_w",
    "prepare_bell",
    "prepare_pauli",
    "dump_matrix",
    "unitary",
    "draw",
]


def flip_to_control(
    control_state: int | list[int], qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:
    r"""Flip qubits from :math:`\ket{\texttt{control_state}}` to :math:`\ket{1\dots1}`.

    The primary usage of this gate is to change the state when controlled applications are applied.
    For instance, all controlled operations are only applied if the control qubits' state is
    :math:`\ket{1}`. This gate is useful for using another state as control.

    Example:

        .. code-block:: python

            from ket import *

            p = Process()
            q = p.alloc(3)

            H(q[:2])

            with around(flip_to_control(0b01), q[:2]):
                ctrl(q[:2], X)(q[2])
    """

    def inner(qubits: Quant) -> Quant:
        if not isinstance(qubits, Quant):
            qubits = reduce(add, qubits)

        length = len(qubits)
        if isinstance(control_state, Sized):
            if len(control_state) != length:
                raise ValueError(
                    f"'to' received a list of length {len(control_state)} to use on {length} qubits"
                )
            state = control_state
        else:
            if length < control_state.bit_length():
                raise ValueError(
                    f"To flip with control_state={control_state} "
                    f"you need at least {control_state.bit_length()} qubits"
                )

            state = [int(i) for i in f"{{:0{length}b}}".format(control_state)]

        for i, qubit in zip(state, qubits):
            if i == 0:
                X(qubit)
        return qubits

    if qubits is None:
        return inner
    return inner(qubits)


def phase_oracle(
    state: int, qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:
    r"""Transform qubits from :math:`\ket{\texttt{state}}` to :math:`-\ket{\texttt{state}}`.

    This gate is useful for marking states in Grover's algorithm.
    """

    def inner(qubits: Quant) -> Quant:
        init, last = qubits[:-1], qubits[-1]
        with around(flip_to_control(state >> 1), init):
            with around(lambda q: X(q) if state & 1 == 0 else None, last):
                ctrl(init, Z)(last)
        return qubits

    if qubits is None:
        return inner
    return inner(qubits)


def prepare_bell(qubit_a: Quant, qubit_b: Quant) -> Quant:
    r"""Prepare a Bell state = :math:`\frac{1}{\sqrt{2}}(\ket{0}+\ket{1})` state."""

    return CNOT(H(qubit_a), qubit_b)


def prepare_ghz(qubits: Quant) -> Quant:
    r"""Prepare a GHZ = :math:`\frac{1}{\sqrt{2}}(\ket{0\dots0}+\ket{1\dots1})` state."""

    ctrl(H(qubits[0]), X)(qubits[1:])

    return qubits


def prepare_w(qubits: Quant) -> Quant:
    r"""Prepare a W = :math:`\frac{1}{\sqrt{n}}\sum_{k=0}^{n}\ket{2^k}` state."""

    size = len(qubits)

    X(qubits[0])
    for i in range(size - 1):
        n = size - i
        ctrl(qubits[i], RY(2 * asin(sqrt((n - 1) / n)))(qubits[i + 1]))
        CNOT(qubits[i + 1], qubits[i])

    return qubits


def dump_matrix(
    gate: Callable, size: int = 1, simulator: Literal["sparse", "dense"] = "sparse"
) -> list[list[complex]]:
    """Get the matrix representation of a quantum gate.

    This function calculates the matrix representation of a quantum gate using
    the specified simulator.

    Args:
        gate: Quantum gate operation to obtain the matrix for.
        size: Number of qubits.
        simulator: Simulator type to use.

    Returns:
        Matrix representation of the quantum gate.
    """

    process = Process(num_qubits=2 * size, execution="batch", simulator=simulator)

    mat = [[0.0j for _ in range(2**size)] for _ in range(2**size)]

    row = process.alloc(size)
    column = process.alloc(size)

    H(column)
    CNOT(column, row)

    gate(row)

    state = dump(column + row)

    for state, amp in state.get().items():
        column = state >> size
        row = state & ((1 << size) - 1)
        mat[row][column] = amp * sqrt(2**size)

    return mat


def prepare_pauli(
    pauli: Literal["X", "Y", "Z"],
    eigenvalue: Literal[+1, -1],
    qubits: Quant | None = None,
) -> Quant | Callable[[Quant], Quant]:
    """Prepare a quantum state in the +1 or -1 eigenstate of a Pauli operator.

    This function prepares a quantum state in the +1 or -1 eigenstate of a specified Pauli operator.
    The resulting quantum state can be obtained by either directly calling the function with qubits,
    or by returning a closure that can be applied to qubits later.

    Args:
        pauli: Pauli operator to prepare the eigenstate for.
        eigenvalue: Eigenvalue of the Pauli operator (+1 or -1).
        qubits: Qubits to prepare the eigenstate on. If None, returns a closure.

    Returns:
        If qubits is provided, returns the resulting quantum state.
        If qubits is None, returns a closure that can be applied to qubits later.
    """

    def inner(qubits: Quant) -> Quant:
        if eigenvalue == +1:
            pass
        elif eigenvalue == -1:
            X(qubits)
        else:
            raise ValueError("Invalid eigenvalue")

        if pauli == "X":
            return H(qubits)
        if pauli == "Y":
            return S(H(qubits))
        if pauli == "Z":
            return qubits
        raise ValueError("Invalid Pauli operator")

    if qubits is None:
        return inner
    return inner(qubits)


def _is_unitary(matrix):
    if len(matrix) != 2 or len(matrix[0]) != 2 or len(matrix[1]) != 2:
        raise ValueError("Input matrix must be a 2x2 matrix")

    conj_transpose = [[matrix[j][i].conjugate() for j in range(2)] for i in range(2)]

    result = [
        [sum(matrix[i][k] * conj_transpose[k][j] for k in range(2)) for j in range(2)]
        for i in range(2)
    ]

    return all(
        isclose(result[i][j], 1 if i == j else 0, abs_tol=1e-10)
        for i in range(2)
        for j in range(2)
    )


def _extract_phase(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    det = a * d - b * c
    return 1 / 2 * atan2(det.imag, det.real)


def _zyz(matrix):
    phase = _extract_phase(matrix)

    matrix = [
        [exp(-1j * phase) * matrix[i][j] for j in range(len(matrix[0]))]
        for i in range(len(matrix))
    ]

    def clip(num):
        """Clip the number to compensate for floating point error"""
        if -(1.0 + 10e-10) < num < 1.0 + 10e-10:
            return min(max(num, -1.0), 1.0)
        raise ValueError("math domain error")

    theta_1 = (
        2 * acos(clip(abs(matrix[0][0])))
        if abs(matrix[0][0]) >= abs(matrix[0][1])
        else 2 * asin(abs(matrix[0][1]))
    ).real

    if not isclose(cos(theta_1 / 2), 0.0, abs_tol=1e-10):
        aux_0_plus_2 = matrix[1][1] / cos(theta_1 / 2)
        theta_0_plus_2 = 2 * atan2(aux_0_plus_2.imag, aux_0_plus_2.real)
    else:
        theta_0_plus_2 = 0.0

    if not isclose(sin(theta_1 / 2), 0.0, abs_tol=1e-10):
        aux_1_sub_2 = matrix[1][0] / sin(theta_1 / 2)
        theta_0_sub_2 = 2 * atan2(aux_1_sub_2.imag, aux_1_sub_2.real)

    else:
        theta_0_sub_2 = 0.0

    theta_0 = (theta_0_plus_2 + theta_0_sub_2) / 2
    theta_2 = (theta_0_plus_2 - theta_0_sub_2) / 2

    return phase, theta_0, theta_1, theta_2


def unitary(matrix: list[list[complex]]) -> Callable[[Quant], Quant]:
    """Create a quantum gate from 2x2 unitary matrix.

    The provided unitary matrix is decomposed into a sequence of rotation gates,
    which together implement an equivalent unitary transformation. When the gate
    is used in a controlled operation, the resulting unitary is equivalent up to
    a global phase.

    Args:
        matrix: Unitary matrix in the format ``[[a, b], [c, d]]``.

    Returns:
        Returns a new callable that implements the unitary operation.

    Raises:
        ValueError: If the input matrix is not unitary.
    """
    if not _is_unitary(matrix):
        raise ValueError("Input matrix is not unitary")

    phase, theta_0, theta_1, theta_2 = _zyz(matrix)

    @global_phase(phase)
    def inner(qubits: Quant):
        RZ(theta_2.real, qubits)
        RY(theta_1.real, qubits)
        RZ(theta_0.real, qubits)
        return qubits

    return inner


def draw(gate: Callable[[Quant], None], num_qubits: int, *, args=(), **kwargs):
    """Draw a quantum gate using Qiskit.

    Args:
        gate: Quantum gate function.
        num_qubits: Number of qubits.
        **kwargs: Keyword arguments to pass to the Qiskit drawer.

    Returns:
        Qiskit circuit diagram of the quantum gate.
    """
    from ..ibm import IBMDevice  # pylint: disable=import-outside-toplevel

    try:
        from qiskit.providers.basic_provider.basic_simulator import (  # pylint: disable=import-outside-toplevel
            BasicSimulator,
        )
    except ImportError as exc:
        raise ImportError(
            "ket.lib.draw requires the qiskit module to be used. You can install them"
            "alongside ket by running `pip install ket[ibm]`."
        ) from exc

    device = IBMDevice(BasicSimulator(), num_qubits, use_qiskit_transpiler=True)
    p = Process(device.configure())
    q = p.alloc(num_qubits)
    gate(*args, q)
    p.execute()
    return device.circuit.draw(**kwargs)
