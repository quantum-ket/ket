# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Grover's search algorithm."""

from math import sqrt, pi
import ket


def grover(size: int, state: int) -> int:
    """Grover's search algorithm.

    Args:
        size (int): The number of qubits to use for the search.
        state (in): The search state.

    Returns:
        int: The measured value from the search.
    """

    p = ket.Process(simulator="dense", num_qubits=size + 1)

    *s, aux = ket.H(p.alloc(size + 1))

    for _ in range(int((pi / 4) * sqrt(2**size))):

        with ket.around(ket.lib.flip_to_control(state), s):
            ket.ctrl(s, ket.RZ(2 * pi))(aux)

        with ket.around(ket.cat(ket.H, ket.X), s):
            ket.ctrl(s[:-1], ket.Z)(s[-1])

    return ket.measure(s).value


if __name__ == "__main__":
    from random import randint

    SIZE = 4
    looking_for = randint(0, pow(2, SIZE) - 1)

    print("Searching for value", looking_for, "using", SIZE, "qubits.")
    result = grover(SIZE, looking_for)
    print("Dense Simulation: measured", result)
    assert result == looking_for
