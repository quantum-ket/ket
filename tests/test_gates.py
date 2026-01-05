# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

from math import pi, sqrt, cos, sin
from cmath import isclose, exp
import ket


GATES = {
    "X": (ket.X, [[0, 1], [1, 0]]),
    "Y": (ket.Y, [[0, -1j], [1j, 0]]),
    "Z": (ket.Z, [[1, 0], [0, -1]]),
    "H": (ket.H, [[1 / sqrt(2), 1 / sqrt(2)], [1 / sqrt(2), -1 / sqrt(2)]]),
    "T": (ket.T, [[1, 0], [0, exp(1j * pi / 4)]]),
    "S": (ket.S, [[1, 0], [0, 1j]]),
    **{
        f"RX({theta})": (
            ket.RX(theta),
            [
                [cos(theta / 2), -1j * sin(theta / 2)],
                [-1j * sin(theta / 2), cos(theta / 2)],
            ],
        )
        for theta in [(2 * pi / i) for i in range(1, 11)]
    },
    **{
        f"RY({theta})": (
            ket.RY(theta),
            [
                [cos(theta / 2), -sin(theta / 2)],
                [sin(theta / 2), cos(theta / 2)],
            ],
        )
        for theta in [(2 * pi / i) for i in range(1, 11)]
    },
    **{
        f"RZ({theta})": (
            ket.RZ(theta),
            [[exp(-1j * theta / 2), 0], [0, exp(1j * theta / 2)]],
        )
        for theta in [(2 * pi / i) for i in range(1, 11)]
    },
}


def validate(gate, simulator, verbose):

    if verbose:
        print(f"{gate=}, {simulator=}")

    p = ket.Process(simulator=simulator)

    gate_func, matrix = GATES[gate]
    gate = ket.qulib.dump_matrix(gate_func, process=p)
    if verbose:
        print(f"{gate[0][0]:+2.4f} {gate[0][1]:+2.4f}")
        print(f"{gate[1][0]:+2.4f} {gate[1][1]:+2.4f}")

    return all(
        isclose(gate[i][j], matrix[i][j], abs_tol=1e-7)
        for i in range(len(matrix))
        for j in range(len(matrix[0]))
    )


def test_gates(verbose=False):
    for simulator in [
        "sparse",
        "dense",
        "dense v2",
        # "dense gpu",
    ]:
        for gate in GATES:
            assert validate(
                gate, simulator, verbose
            ), f"assert fail {gate=}, {simulator=}"


if __name__ == "__main__":
    test_gates(True)
    print("OK")
