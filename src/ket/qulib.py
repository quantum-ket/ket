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
from collections.abc import Sized, Iterable

from .base import Quant, Process
from .operations import ctrl, around, dump
from .gates import RZ, X, Z, H, RY, CNOT, S, global_phase

try:
    from IPython import get_ipython

    IN_NOTEBOOK = get_ipython().__class__.__name__ == "ZMQInteractiveShell"
except ImportError:
    IN_NOTEBOOK = False

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
    gate: Callable,
    num_qubits: int | list[int] = 1,
    args=(),
    process: Process | None = None,
) -> list[list[complex]]:
    """Get the matrix representation of a quantum gate.

    This function calculates the matrix representation of a quantum gate.

    Args:
        gate: Quantum gate operation to obtain the matrix for.
        num_qubits: Number of qubits.
        args: Classical arguments to pass to the gate function.
        process: Quantum process used to generate the matrix.

    Returns:
        Matrix representation of the quantum gate.
    """

    if isinstance(num_qubits, Iterable):
        qubit_args = num_qubits
    else:
        qubit_args = [num_qubits]

    num_qubits = sum(num_qubits) if isinstance(num_qubits, Iterable) else num_qubits

    if process is None:
        process = Process(
            num_qubits=2 * num_qubits,
            execution="batch",
            simulator="sparse",
        )

    mat = [[0.0j for _ in range(2**num_qubits)] for _ in range(2**num_qubits)]

    qubit_args = [process.alloc(n) for n in qubit_args]
    row = reduce(add, qubit_args)
    column = process.alloc(num_qubits)

    H(column)
    CNOT(column, row)

    gate(*args, *qubit_args)

    state = dump(column + row)

    for state, amp in state.get().items():
        column = state >> num_qubits
        row = state & ((1 << num_qubits) - 1)
        mat[row][column] = amp * sqrt(2**num_qubits)

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


DRAW_STYLE = {
    "textcolor": "#0f1f2e",
    "gatetextcolor": "#f4f8ff",
    "subtextcolor": "#f4f8ff",
    "linecolor": "#0f1f2e",
    "creglinecolor": "#6b7682",
    "gatefacecolor": "#0f7cfb",
    "barrierfacecolor": "#c6cdd5",
    "backgroundcolor": "#f4f8ff",
    "displaytext": {
        "u1": "P",
        "rx": "RX",
        "ry": "RY",
        "rz": "RZ",
    },
    "displaycolor": {
        "u1": ["#008699", "#f4f8ff"],
        "x": ["#0f7cfb", "#f4f8ff"],
        "y": ["#0c63c9", "#f4f8ff"],
        "z": ["#009cb3", "#f4f8ff"],
        "h": ["#3f96fc", "#f4f8ff"],
        "rx": ["#063264", "#f4f8ff"],
        "ry": ["#094a97", "#f4f8ff"],
        "rz": ["#008699", "#f4f8ff"],
        "cx": ["#0f7cfb", "#f4f8ff"],
        "ccx": ["#0f7cfb", "#f4f8ff"],
        "mcx": ["#0f7cfb", "#f4f8ff"],
        "cy": ["#0c63c9", "#f4f8ff"],
        "cz": ["#009cb3", "#f4f8ff"],
    },
}


def draw(
    gate: Callable[[Quant], None],
    num_qubits: int | list[int],
    *,
    args=(),
    decompose: bool = False,
    title: str | None = None,
    **kwargs,
):
    """Draw a quantum gate using Qiskit.

    Args:
        gate: Quantum gate function.
        num_qubits: Number of qubits.
        args: Classical arguments to pass to the gate function.
        decompose: Decompose controlled gates. Defaults to False.
        **kwargs: Keyword arguments to pass to the Qiskit drawer.

    Returns:
        Qiskit circuit diagram of the quantum gate.
    """
    from .ibm import IBMDeviceForDraw  # pylint: disable=import-outside-toplevel

    device = IBMDeviceForDraw(
        sum(num_qubits) if isinstance(num_qubits, Iterable) else num_qubits,
        decompose,
    )
    p = Process(device.config())
    if isinstance(num_qubits, Iterable):
        q = [p.alloc(n) for n in num_qubits]
    else:
        q = [p.alloc(num_qubits)]
    gate(*args, *q)
    p.execute()

    if "output" not in kwargs and IN_NOTEBOOK:
        kwargs["output"] = "mpl"
    if "style" not in kwargs:
        kwargs["style"] = DRAW_STYLE

    fig = device.circuit.draw(**kwargs)
    if title is not None:
        fig.suptitle(title)
    return fig
