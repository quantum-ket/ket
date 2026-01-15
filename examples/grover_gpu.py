# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Grover's search algorithm."""

from random import randint
from math import sqrt, pi
from time import time
import plotly.express as px
import ket


def grover(size: int, oracle, outcomes: int = 1, simulator: str = "dense") -> int:
    """Grover's search algorithm.

    Args:
        size (int): The number of qubits to use for the search.
        oracle (Callable): The oracle function to use for the search.
        outcomes (int): The number of outcomes to search for.
        simulator (str): The simulator to use for the process.

    Returns:
        int: The measured value from the search.

    """

    p = ket.Process(simulator=simulator, num_qubits=size, execution="batch")

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

    return ket.measure(s).get()  # Measure the final state and return the result.


if __name__ == "__main__":

    data = {
        "size": [],
        "simulator": [],
        "time": [],
    }

    for size in range(4, 20):
        looking_for = randint(0, pow(2, size) - 1)
        for simulator in ["dense", "dense gpu"]:
            start = time()
            result = grover(
                size,
                ket.qulib.oracle.phase_oracle(looking_for),
                simulator=simulator,
            )
            end = time()
            assert result == looking_for, "Grover algorithm failed!"
            data["size"].append(size)
            data["simulator"].append(simulator)
            data["time"].append(end - start)
            print(f"{simulator.upper():<9}: {end - start:3.4f}s.")

    fig = px.line(
        data,
        x="size",
        y="time",
        log_y=True,
        color="simulator",
        title="Grover's Algorithm Simulation Time",
        labels={"size": "Number of Qubits", "time": "Time (s)"},
    )
    fig.show()
