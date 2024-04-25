# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

from math import pi, sqrt
from cmath import isclose, exp
import ket
import pytest


def validate(matrix):
    from ket.lib import _extract_phase

    gate = ket.lib.dump_matrix(ket.lib.unitary(matrix))

    phase = _extract_phase(matrix)
    matrix = [
        [exp(-1j * phase) * matrix[i][j] for j in range(len(matrix[0]))]
        for i in range(len(matrix))
    ]

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


def test_unitary():
    for gate in [X, Y, Z, H, T, S]:
        assert validate(gate)


def test_not_unitary():
    with pytest.raises(ValueError):
        sqrt_2 = 1 / sqrt(2)
        ket.lib.unitary([[sqrt_2, sqrt_2], [sqrt_2, sqrt_2]])


def test_not_2x2_matrix():
    with pytest.raises(ValueError):
        ket.lib.unitary([[1, 2], [4, 5, 6]])


def make_ctrl_gate(matrix):
    return [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, matrix[0][0], matrix[0][1]],
        [0, 0, matrix[1][0], matrix[1][1]],
    ]


def ctrl_validate(matrix):
    gate = lambda q: ket.ctrl(q[0], ket.lib.unitary(matrix))(q[1])
    gate = ket.lib.dump_matrix(gate, size=2)

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
    test_ctrl_unitary()
    print("OK")
