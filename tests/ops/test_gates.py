# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from math import pi, sqrt, cos, sin
from cmath import exp
from itertools import product
import ket
from ket.qulib import dump_matrix
from ket.qulib.gates import unitary

# --- Base Mathematical Definitions ---

GATES = {
    "X": (ket.X, [[0, 1], [1, 0]]),
    "Y": (ket.Y, [[0, -1j], [1j, 0]]),
    "Z": (ket.Z, [[1, 0], [0, -1]]),
    "H": (ket.H, [[1 / sqrt(2), 1 / sqrt(2)], [1 / sqrt(2), -1 / sqrt(2)]]),
    "T": (ket.T, [[1, 0], [0, exp(1j * pi / 4)]]),
    "S": (ket.S, [[1, 0], [0, 1j]]),
}


def rx_matrix(theta):
    return [
        [cos(theta / 2), -1j * sin(theta / 2)],
        [-1j * sin(theta / 2), cos(theta / 2)],
    ]


def ry_matrix(theta):
    return [
        [cos(theta / 2), -sin(theta / 2)],
        [sin(theta / 2), cos(theta / 2)],
    ]


def rz_matrix(theta):
    return [[exp(-1j * theta / 2), 0], [0, exp(1j * theta / 2)]]


def u3_matrix(theta: float, phi: float, lambda_: float) -> list:
    return [
        [
            exp(-1j * (phi + lambda_) / 2) * cos(theta / 2),
            -exp(-1j * (phi - lambda_) / 2) * sin(theta / 2),
        ],
        [
            exp(1j * (phi - lambda_) / 2) * sin(theta / 2),
            exp(1j * (phi + lambda_) / 2) * cos(theta / 2),
        ],
    ]


def validate_matrix(dumped_matrix, expected_matrix, tol=1e-5):
    """Helper to validate if two matrices are equivalent using NumPy."""
    return np.allclose(dumped_matrix, expected_matrix, atol=tol)


# --- Tests ---


@pytest.mark.parametrize("simulator", ["sparse", "sparse v2", "dense", "dense v2"])
@pytest.mark.parametrize("gate_name", GATES.keys())
def test_standard_gates(simulator, gate_name):
    """Test standard gates against their theoretical unitary matrices across simulators."""
    p = ket.Process(simulator=simulator)
    gate_func, expected_matrix = GATES[gate_name]

    dumped_matrix = dump_matrix(gate_func, process=p)
    assert validate_matrix(dumped_matrix, expected_matrix)


@pytest.mark.parametrize("angle", [2 * pi / i for i in range(1, 11)])
def test_rotation_gates(angle):
    """Test rotation gates (RX, RY, RZ) dynamically across various angles."""
    assert validate_matrix(dump_matrix(ket.RX(angle)), rx_matrix(angle))
    assert validate_matrix(dump_matrix(ket.RY(angle)), ry_matrix(angle))
    assert validate_matrix(dump_matrix(ket.RZ(angle)), rz_matrix(angle))


def test_u3_gate_parameter_grid():
    """Test U3 gate across a grid of theta, phi, and lambda parameters."""
    angles = [0, pi / 2, pi, 2 * pi]

    for theta, phi, lambda_ in product(angles, repeat=3):
        # Dump U3 matrix and compare
        gate_mat = dump_matrix(ket.U3(theta, phi, lambda_))
        expected_mat = u3_matrix(theta, phi, lambda_)
        assert validate_matrix(gate_mat, expected_mat)


def test_custom_unitary_gate_valid():
    """Test the custom unitary gate builder with valid matrices."""
    # Test building X and H from matrices
    custom_x = unitary(GATES["X"][1], up_to_global_phase=True)
    custom_h = unitary(GATES["H"][1], up_to_global_phase=True)

    assert validate_matrix(dump_matrix(custom_x), GATES["X"][1])
    assert validate_matrix(dump_matrix(custom_h), GATES["H"][1])


def test_custom_unitary_gate_invalid():
    """Test if the unitary gate builder rejects non-unitary or malformed matrices."""
    # Not unitary
    with pytest.raises(ValueError, match="Input matrix is not unitary"):
        unitary([[1, 1], [1, 1]])

    # Invalid shape
    with pytest.raises(ValueError, match="Input matrix must be a 2x2 matrix"):
        unitary([[1, 2], [3, 4, 5]])
