from math import pi, sqrt
from cmath import isclose, exp
import ket


def validate(matrix):
    from ket.gates import _extract_phase

    gate = ket.lib.dump_matrix(ket.unitary(matrix))

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


def test_unitary():
    X = [[0, 1], [1, 0]]
    Y = [[0, -1j], [1j, 0]]
    Z = [[1, 0], [0, -1]]
    H = [[1 / sqrt(2), 1 / sqrt(2)], [1 / sqrt(2), -1 / sqrt(2)]]
    T = [[1, 0], [0, exp(1j * pi / 4)]]
    S = [[1, 0], [0, 1j]]
    for gate in [X, Y, Z, H, T, S]:
        assert validate(gate)


if __name__ == "__main__":
    test_unitary()
    print("OK")
