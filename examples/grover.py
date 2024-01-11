# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Grover's search algorithm."""

from math import sqrt, pi
from typing import Callable
import ket


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
        with ket.around(ket.H, s):
            ket.phase_oracle(0, s)  # Apply the diffusor function to the state.

    return ket.measure(s).value  # Measure the final state and return the result.


if __name__ == "__main__":
    from random import randint

    SIZE = 12
    looking_for = randint(0, pow(2, SIZE) - 1)

    print("Searching for value", looking_for, "using", SIZE, "qubits.")

    print("Dense Simulation: measured", grover(SIZE, ket.phase_oracle(looking_for)))
