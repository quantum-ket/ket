"""Quantum gate construction."""

from __future__ import annotations

# SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from itertools import starmap
from typing import Callable
from cmath import isclose, phase as cmath_phase, exp
from math import atan2
from ..gates import X, P, RZ, RY, CNOT, global_phase
from ..operations import adj, around
from ..base import Quant

__all__ = [
    "unitary",
    "diagonal",
    "ucz",
]


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


def _zyz(matrix):
    a, b = matrix[0]
    c, d = matrix[1]

    det = cmath_phase(a * d - b * c)
    phase = det / 2

    theta = 2 * atan2(abs(c), abs(a))

    ang1 = cmath_phase(d)
    ang2 = cmath_phase(c)

    phi = ang1 + ang2 - det
    lam = ang1 - ang2

    return phase, phi, theta, lam


def _apply_global_phase(phase: float, qubits: Quant):
    P(phase, qubits[0])
    with around(X, qubits[0]):
        P(phase, qubits[0])


def unitary(
    matrix: list[list[complex]], up_to_global_phase: bool = False
) -> Callable[[Quant], Quant]:
    """Create a quantum gate from 2x2 unitary matrix.

    The provided unitary matrix is decomposed into a sequence of rotation gates,
    which together implement an equivalent unitary transformation. When the gate
    is used in a controlled operation, the resulting unitary is equivalent up to
    a global phase.

    Args:
        matrix: Unitary matrix in the format ``[[a, b], [c, d]]``.
        up_to_global_phase: If True, the resulting unitary is equivalent up to
            a global phase even when the gate is not used in a controlled
            operation.

    Returns:
        Returns a new callable that implements the unitary operation.

    Raises:
        ValueError: If the input matrix is not unitary.
    """
    if not _is_unitary(matrix):
        raise ValueError("Input matrix is not unitary")

    phase, phi, theta, lam = _zyz(matrix)

    @global_phase(0.0 if up_to_global_phase else phase)
    def inner(qubits: Quant):
        if up_to_global_phase:
            _apply_global_phase(phase, qubits)
        RZ(lam.real, qubits)
        RY(theta.real, qubits)
        RZ(phi.real, qubits)
        return qubits

    return inner


def ucz(angles, qubits, *, _last=True):
    """Uniformly controlled z-axis rotation gate.

    Implements a uniformly controlled rotation around the z-axis, as proposed
    in https://arxiv.org/abs/quant-ph/0406176.

    Args:
        angles: List of rotation angles.
        qubits: List of qubits to apply the gate to.
    """
    if len(angles) != 1 << (len(qubits) - 1):
        raise ValueError(
            f"Invalid number of angles. Expected {1 << (len(qubits) - 1)}, got {len(angles)}."
        )
    if len(qubits) == 1:
        RZ(angles[0], qubits)
        return

    h = len(angles) // 2
    uh = angles[:h]
    lh = angles[h:]

    angles1 = list(starmap(lambda a, b: (a + b) / 2, zip(lh, uh)))
    angles2 = list(starmap(lambda a, b: (a - b) / 2, zip(lh, uh)))

    ucz(angles1, qubits[1:], _last=False)
    CNOT(qubits[0], qubits[-1])
    adj(ucz)(angles2, qubits[1:], _last=False)
    if _last:
        CNOT(qubits[0], qubits[-1])


def _diagonal_gate(angles, qubits, up_to_global_phase=True):
    if len(angles) == 1:
        return angles[0]

    n = len(angles)

    rz_angle = [
        (exp(1j * angles[2 * i]), exp(1j * angles[2 * i + 1])) for i in range(n // 2)
    ]

    rz_fix = [cmath_phase(a[0] * a[1]) / 2 for a in rz_angle]

    rz_angle = list(starmap(lambda r, a: r[0] * exp(-a * 1j), zip(rz_angle, rz_fix)))

    rz_angle = [-cmath_phase(a) * 2 for a in rz_angle]

    ucz(rz_angle, qubits)

    phase = _diagonal_gate(rz_fix, qubits[:-1], False)

    if up_to_global_phase:
        _apply_global_phase(phase, qubits)
    return phase


def diagonal(*angles, up_to_global_phase=True):
    r"""Create a diagonal quantum gate.

    .. math::

        U = \sum_i \theta_i \left| i \right> \left< i \right|

    Args:
        angles: List of rotation angles :math:`\theta_i`.
        up_to_global_phase: If True, the resulting unitary is equivalent up to
            a global phase.
    """

    if len(angles).bit_count() != 1:
        raise ValueError("Invalid number of angles. Expected a power of 2.")

    return partial(_diagonal_gate, angles, up_to_global_phase=up_to_global_phase)
