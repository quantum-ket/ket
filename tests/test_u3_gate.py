# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

from itertools import product
import math
import cmath
import ket
from ket import qulib

def u3_matrix(theta: float, phi: float, lambda_: float) -> list:
    return [
        [
            cmath.exp(-1j * (phi + lambda_) / 2) * math.cos(theta / 2),
            -cmath.exp(-1j * (phi - lambda_) / 2) * math.sin(theta / 2),
        ],
        [
            cmath.exp(1j * (phi - lambda_) / 2) * math.sin(theta / 2),
            cmath.exp(1j * (phi + lambda_) / 2) * math.cos(theta / 2),
        ],
    ]


def linspace(start, stop, num):
    yield start
    step = (stop - start) / (num - 1)
    next_num = start
    for _ in range(num - 1):
        next_num += step
        yield next_num


def test_u3_0_2pi_gate():
    for theta, phi, lambda_ in product(linspace(0, 2 * math.pi, 10), repeat=3):
        gate = qulib.dump_matrix(ket.U3(theta, phi, lambda_))
        matrix = u3_matrix(theta, phi, lambda_)

        assert all(
            cmath.isclose(gate[i][j], matrix[i][j], abs_tol=1e-7)
            for i in range(2)
            for j in range(2)
        )


def test_u3_m2pi_0_gate():
    for theta, phi, lambda_ in product(linspace(-2 * math.pi, 0.0, 10), repeat=3):
        gate = qulib.dump_matrix(lambda q: ket.U3(theta, phi, lambda_, q))
        matrix = u3_matrix(theta, phi, lambda_)

        assert all(
            cmath.isclose(gate[i][j], matrix[i][j], abs_tol=1e-7)
            for i in range(2)
            for j in range(2)
        )


if __name__ == "__main__":
    test_u3_0_2pi_gate()
    test_u3_m2pi_0_gate()
    print("OK")
