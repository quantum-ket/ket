# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

from math import pi, sqrt, cos, sin
from cmath import isclose, exp
import ket
import pytest


def validate(matrix):
    gate = ket.qulib.dump_matrix(ket.qulib.gates.unitary(matrix, True))

    return all(
        isclose(gate[i][j], matrix[i][j], abs_tol=1e-10)
        for i in range(len(matrix))
        for j in range(len(matrix[0]))
    )


X = [[0, 1], [1, 0]]
Y = [[0, -1j], [1j, 0]]
Z = [[1, 0], [0, -1]]
H = [[1 / sqrt(2), 1 / sqrt(2)], [1 / sqrt(2), -1 / sqrt(2)]]
T = [[1, 0], [0, exp(1j * pi / 4)]]
S = [[1, 0], [0, 1j]]
RX = lambda theta: [
    [cos(theta / 2), -1j * sin(theta / 2)],
    [-1j * sin(theta / 2), cos(theta / 2)],
]
RY = lambda theta: [
    [cos(theta / 2), -sin(theta / 2)],
    [sin(theta / 2), cos(theta / 2)],
]
RZ = lambda theta: [[exp(-1j * theta / 2), 0], [0, exp(1j * theta / 2)]]


def test_unitary():
    for gate in [
        X,
        Y,
        Z,
        H,
        T,
        S,
        *[RX(2 * pi / i) for i in range(1, 11)],
        *[RY(2 * pi / i) for i in range(1, 11)],
        *[RZ(2 * pi / i) for i in range(1, 11)],
    ]:
        assert validate(gate)


def test_not_unitary():
    with pytest.raises(ValueError):
        sqrt_2 = 1 / sqrt(2)
        ket.qulib.gates.unitary([[sqrt_2, sqrt_2], [sqrt_2, sqrt_2]])


def test_not_2x2_matrix():
    with pytest.raises(ValueError):
        ket.qulib.gates.unitary([[1, 2], [4, 5, 6]])


def make_ctrl_gate(matrix):
    return [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, matrix[0][0], matrix[0][1]],
        [0, 0, matrix[1][0], matrix[1][1]],
    ]


def ctrl_validate(matrix):
    gate = lambda q: ket.ctrl(q[0], ket.qulib.gates.unitary(matrix))(q[1])
    gate = ket.qulib.dump_matrix(gate, num_qubits=2)

    matrix = make_ctrl_gate(matrix)

    return all(
        isclose(gate[i][j], matrix[i][j], abs_tol=1e-10)
        for i in range(len(matrix))
        for j in range(len(matrix[0]))
    )


def test_ctrl_unitary():
    for gate in [X, Y, Z, H, T, S]:
        assert ctrl_validate(gate)


if __name__ == "__main__":
    test_unitary()
    print("OK")
