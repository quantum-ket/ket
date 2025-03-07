# SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

from math import sqrt, pi, cos, sin
from cmath import exp, isclose
import ket
from ket import qulib

GATES = {
    ket.X: [[0, 1], [1, 0]],
    ket.Y: [[0, -1j], [1j, 0]],
    ket.Z: [[1, 0], [0, -1]],
    ket.H: [[1 / sqrt(2), 1 / sqrt(2)], [1 / sqrt(2), -1 / sqrt(2)]],
    ket.T: [[1, 0], [0, exp(1j * pi / 4)]],
    ket.S: [[1, 0], [0, 1j]],
}

ROTATION_GATES = {
    ket.RX: lambda theta: [
        [cos(theta / 2), -1j * sin(theta / 2)],
        [-1j * sin(theta / 2), cos(theta / 2)],
    ],
    ket.RY: lambda theta: [
        [cos(theta / 2), -sin(theta / 2)],
        [sin(theta / 2), cos(theta / 2)],
    ],
    ket.RZ: lambda theta: [[exp(-1j * theta / 2), 0], [0, exp(1j * theta / 2)]],
}


def linspace(start, stop, num):
    yield start
    step = (stop - start) / (num - 1)
    next_num = start
    for _ in range(num - 1):
        next_num += step
        yield next_num


def test_decomposition_su2():
    n = 7

    for gate, mat_gate in ROTATION_GATES.items():
        for ang in linspace(0.0, 2 * pi, 8):
            matrix = mat_gate(ang)

            ctrl_gate = lambda q: ket.ctrl(q[:-1], gate(ang))(q[-1])
            result_matrix = ket.qulib.dump_matrix(ctrl_gate, num_qubits=n)

            gate_matrix = [
                [result_matrix[-2][-2], result_matrix[-2][-1]],
                [result_matrix[-1][-2], result_matrix[-1][-1]],
            ]

            eye_matrix = result_matrix

            eye_matrix[-2][-2] = 1
            eye_matrix[-2][-1] = 0
            eye_matrix[-1][-2] = 0
            eye_matrix[-1][-1] = 1

            eye = list(
                list(1 if i == j else 0 for i in range(2**n)) for j in range(2**n)
            )

            assert all(
                isclose(gate_matrix[i][j], matrix[i][j], abs_tol=1e-10)
                for i in range(2)
                for j in range(2)
            )

            assert all(
                isclose(eye_matrix[i][j], eye[i][j], abs_tol=1e-10)
                for i in range(2**n)
                for j in range(2**n)
            )


def test_decomposition_c_t():

    n = 7

    for ket_gate, matrix in GATES.items():
        gate = lambda q: ket.ctrl(q[:-1], ket_gate)(q[-1])
        result_matrix = ket.qulib.dump_matrix(gate, num_qubits=n)

        gate_matrix = [
            [result_matrix[-2][-2], result_matrix[-2][-1]],
            [result_matrix[-1][-2], result_matrix[-1][-1]],
        ]

        eye_matrix = result_matrix

        eye_matrix[-2][-2] = 1
        eye_matrix[-2][-1] = 0
        eye_matrix[-1][-2] = 0
        eye_matrix[-1][-1] = 1

        eye = list(list(1 if i == j else 0 for i in range(2**n)) for j in range(2**n))

        assert all(
            isclose(gate_matrix[i][j], matrix[i][j], abs_tol=1e-10)
            for i in range(2)
            for j in range(2)
        )

        assert all(
            isclose(eye_matrix[i][j], eye[i][j], abs_tol=1e-10)
            for i in range(2**n)
            for j in range(2**n)
        )


def test_decomposition_t_c():
    n = 7

    for ket_gate in GATES.keys():
        gate = lambda q: ket.ctrl(q[1:], ket_gate)(q[0])

        decompose_matrix = qulib.dump_matrix(gate, num_qubits=n)

        not_decompose_matrix = qulib.dump_matrix(gate, num_qubits=n)

        assert all(
            isclose(decompose_matrix[i][j], not_decompose_matrix[i][j], abs_tol=1e-10)
            for i in range(2**n)
            for j in range(2**n)
        )


if __name__ == "__main__":
    test_decomposition_c_t()
    # test_decomposition_t_c()
    # test_decomposition_is_enabled()
    # test_decomposition_su2()
    print("Ok")
