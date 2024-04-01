from math import sqrt, pi
from cmath import exp, isclose
from pprint import pprint
import ket
import numpy as np

GATES = {
    ket.X: [[0, 1], [1, 0]],
    ket.Y: [[0, -1j], [1j, 0]],
    ket.Z: [[1, 0], [0, -1]],
    ket.H: [[1 / sqrt(2), 1 / sqrt(2)], [1 / sqrt(2), -1 / sqrt(2)]],
    ket.T: [[1, 0], [0, exp(1j * pi / 4)]],
    ket.S: [[1, 0], [0, 1j]],
}


def test_decomposition():
    ket.set_default_process_configuration(decompose=True, force_configuration=True)

    n = 7

    for ket_gate, matrix in GATES.items():
        gate = lambda q: ket.ctrl(q[:-1], ket_gate)(q[-1])
        result_matrix = ket.lib.dump_matrix(gate, size=n)

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


def test_decomposition_is_enabled():
    n = 5

    ket.set_default_process_configuration(decompose=False, force_configuration=True)

    p1 = ket.Process()
    q = p1.alloc(5)
    ket.ctrl(q[:-1], ket.H)(q[-1])

    ket.set_default_process_configuration(decompose=True, force_configuration=True)

    p2 = ket.Process()
    q = p2.alloc(5)
    ket.ctrl(q[:-1], ket.H)(q[-1])

    assert len(p1.get_instructions()) < len(p2.get_instructions())


if __name__ == "__main__":
    test_decomposition()
    test_decomposition_is_enabled()
    print("Ok")
