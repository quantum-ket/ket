# SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

from ket import X, Y, Z, H, T, S, RX, RY, RZ, C, Process
from ket.qulib import dump_matrix
import numpy as np

GATES = {
    X: [[0, 1], [1, 0]],
    Y: [[0, -1j], [1j, 0]],
    Z: [[1, 0], [0, -1]],
    H: [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]],
    T: [[1, 0], [0, np.exp(1j * np.pi / 4)]],
    S: [[1, 0], [0, 1j]],
}

ROTATION_GATES = {
    RX: lambda theta: [
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)],
    ],
    RY: lambda theta: [
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2), np.cos(theta / 2)],
    ],
    RZ: lambda theta: [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
}


def validate(result, matrix):
    """Check resulting gate matrix"""

    gate_matrix = [
        [result[-2][-2], result[-2][-1]],
        [result[-1][-2], result[-1][-1]],
    ]

    eye_matrix = result

    eye_matrix[-2][-2] = 1
    eye_matrix[-2][-1] = 0
    eye_matrix[-1][-2] = 0
    eye_matrix[-1][-1] = 1

    return np.allclose(eye_matrix, np.eye(len(eye_matrix))) and np.allclose(
        gate_matrix, matrix
    )


def base_test_su2(c, a):
    """Test the decomposition of Rotation gates."""
    n = 2 * (c + 1) + a

    assert all(
        validate(
            dump_matrix(
                C(gate(angle)),
                num_qubits=(c, 1),
                process=Process(
                    num_qubits=n,
                    coupling_graph=[(i, j) for i in range(n) for j in range(i)],
                    simulator="sparse",
                ),
            ),
            matrix(angle),
        )
        for gate, matrix in ROTATION_GATES.items()
        for angle in np.linspace(0.0, 2 * np.pi, 8)
    )


def base_test_u2(c, a):
    """Test the decomposition of U2 gates."""
    n = 2 * (c + 1) + a

    assert all(
        validate(
            dump_matrix(
                C(gate),
                num_qubits=(c, 1),
                process=Process(
                    num_qubits=n,
                    coupling_graph=[(i, j) for i in range(n) for j in range(i)],
                    simulator="sparse",
                ),
            ),
            matrix,
        )
        for gate, matrix in GATES.items()
    )


def test_u2():
    """Test the decomposition of U2 gates up to 6 qubits."""
    n = 6
    for c in range(1, n):
        for a in range(c):
            base_test_u2(c, a)


def test_su2():
    """Test the decomposition of SU2 gates up to 6 qubits."""
    n = 6
    for c in range(1, n):
        for a in range(c):
            base_test_su2(c, a)


if __name__ == "__main__":
    test_su2()
    test_u2()
    print("OK")

