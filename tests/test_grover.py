# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Grover's search algorithm."""

from math import sqrt, pi
from typing import Callable
import ket
from random import randint


def grover(size: int, oracle: Callable, outcomes: int = 1) -> int:
    """Grover's search algorithm.

    Args:
        size (int): The number of qubits to use for the search.
        oracle (Callable): The oracle function to use for the search.
        outcomes (int): The number of outcomes to search for.

    Returns:
        int: The measured value from the search.

    """

    p = ket.Process(simulator="dense", num_qubits=size)

    s = ket.H(
        p.alloc(size)
    )  # Initialize the state to a superposition of all possible outcomes.

    steps = int(
        (pi / 4) * sqrt(2**size / outcomes)
    )  # Calculate the number of iterations for the algorithm.
    for _ in range(steps):
        oracle(s)  # Apply the oracle function to the state.
        with ket.around(ket.cat(ket.H, ket.X), s):
            ket.CZ(*s)  # Apply the diffusor function to the state.

    return ket.measure(s).value  # Measure the final state and return the result.


def test_grover():
    SIZE = 12
    looking_for = randint(0, pow(2, SIZE) - 1)
    assert looking_for == grover(SIZE, ket.qulib.oracle.phase_oracle(looking_for))


if __name__ == "__main__":
    test_grover()
    print("OK")
