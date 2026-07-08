# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from math import pi, sqrt
from cmath import isclose, exp
import ket
from ket import Process, ctrl, using_aux, around, kron, C, X, H
from ket.qulib import dump_matrix
from ket.qulib.gates import unitary

# Reuse gate definitions from test_gates conceptually
GATES = {
    "X": (ket.X, [[0, 1], [1, 0]]),
    "Y": (ket.Y, [[0, -1j], [1j, 0]]),
    "Z": (ket.Z, [[1, 0], [0, -1]]),
    "H": (ket.H, [[1 / sqrt(2), 1 / sqrt(2)], [1 / sqrt(2), -1 / sqrt(2)]]),
    "T": (ket.T, [[1, 0], [0, exp(1j * pi / 4)]]),
    "S": (ket.S, [[1, 0], [0, 1j]]),
}


def make_ctrl_matrix(matrix):
    """Generates a 4x4 controlled-U matrix from a 2x2 U matrix."""
    return [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, matrix[0][0], matrix[0][1]],
        [0, 0, matrix[1][0], matrix[1][1]],
    ]


@pytest.mark.parametrize("gate_name", GATES.keys())
def test_ctrl_standard_gates(gate_name):
    """Test that the `ctrl` modifier correctly expands 2x2 gates into 4x4 matrices."""
    gate_func, matrix = GATES[gate_name]
    expected_ctrl_matrix = make_ctrl_matrix(matrix)

    # Use C() wrapper which uses ctrl() under the hood
    dumped_matrix = dump_matrix(C(gate_func), num_qubits=(1, 1))

    # Validate equality utilizing NumPy
    assert np.allclose(dumped_matrix, expected_ctrl_matrix, atol=1e-7)


def test_ctrl_custom_unitary():
    """Test if a custom unitary matrix is controlled correctly."""
    custom_matrix = [[0, 1], [1, 0]]  # Pauli X
    expected_ctrl = make_ctrl_matrix(custom_matrix)

    # Lambda wrapper to apply control to the custom unitary
    gate = lambda q: ctrl(q[0], unitary(custom_matrix))(q[1])
    dumped_matrix = dump_matrix(gate, num_qubits=2)

    assert np.allclose(dumped_matrix, expected_ctrl, atol=1e-7)


# --- Advanced Decompositions Stress Tests ---


def validate_decomposition(result_matrix, expected_base_matrix):
    """Check if the extracted top-level target bits match the expected 2x2 matrix.

    Because auxiliary qubits enlarge the matrix, we isolate the target subsystem.
    """
    gate_matrix = [
        [result_matrix[-2][-2], result_matrix[-2][-1]],
        [result_matrix[-1][-2], result_matrix[-1][-1]],
    ]

    eye_matrix = result_matrix.copy()
    eye_matrix[-2][-2] = 1
    eye_matrix[-2][-1] = 0
    eye_matrix[-1][-2] = 0
    eye_matrix[-1][-1] = 1

    return np.allclose(eye_matrix, np.eye(len(eye_matrix))) and np.allclose(
        gate_matrix, expected_base_matrix
    )


@using_aux(a=lambda c: 0 if len(c) <= 2 else 1)
def v_chain_clean(c, t, a):
    """V Chain decomposition with safe clean auxiliary management."""
    if len(c) <= 2:
        ctrl(c, X)(t)
    else:
        with around(ctrl(c[:2], X), a):
            v_chain_clean(a + c[2:], t)


def test_v_chain_clean_decomposition():
    """Test V-chain decomposition representing a multi-controlled NOT using clean aux."""
    c = 6
    a = c - 2
    n = 2 * (c + 1) + a

    dumped = dump_matrix(
        v_chain_clean,
        num_qubits=(c, 1),
        process=Process(
            num_qubits=n,
            simulator="sparse",
            execution="live",
        ),
    )
    assert validate_decomposition(dumped, GATES["X"][1])


@using_aux(unsafe=True, a=lambda c: len(c) - 2)
def v_chain_dirty(c, t, a):
    """V Chain decomposition with unsafe dirty auxiliary management."""

    def inner(c, t, a, skip=False):
        if len(c) <= 2:
            ctrl(c, X)(t)
        elif skip:
            inner(c=c[:-1], t=a[-1], a=a[:-1])
        else:
            with around(ctrl(c[-1] + a[-1], X), t):
                inner(c=c[:-1], t=a[-1], a=a[:-1])

    inner(c, t, a)
    inner(c, t, a, True)


def test_v_chain_dirty_decomposition():
    """Test V-chain decomposition representing a multi-controlled NOT using dirty aux."""
    c = 6
    a = c - 2
    n = 2 * (c + 1) + a

    dumped = dump_matrix(
        v_chain_dirty,
        num_qubits=(c, 1),
        process=Process(
            num_qubits=n,
            simulator="sparse",
            execution="live",
        ),
    )
    assert validate_decomposition(dumped, GATES["X"][1])
