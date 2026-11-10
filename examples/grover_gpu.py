# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Grover's search: CPU vs GPU simulation time.

Requires: a GPU, and pip install ket-lang[plot]
"""

from math import pi, sqrt
from random import randrange
from time import perf_counter

from plotly import express as px

from ket import *


def grover(n: int, target: int, simulator: str) -> int:
    """Return the state found by Grover's search on `simulator`."""
    p = Process(simulator=simulator, num_qubits=n, execution="batch")
    q = H(p.alloc(n))
    for _ in range(int(pi / 4 * sqrt(2**n))):
        qulib.oracle.phase_oracle(target, q)
        with around(cat(H, X), q):
            CZ(*q)
    # `measure` needs live execution; in batch, sample a single shot.
    return sample(q, shots=1).most_frequent_state()


data = {"qubits": [], "simulator": [], "time": []}
for n in range(4, 20):
    target = randrange(2**n)
    for simulator in ["dense", "dense gpu"]:
        start = perf_counter()
        assert grover(n, target, simulator) == target
        elapsed = perf_counter() - start
        print(f"{n:2} qubits, {simulator:9}: {elapsed:.4f} s")
        data["qubits"].append(n)
        data["simulator"].append(simulator)
        data["time"].append(elapsed)

px.line(data, x="qubits", y="time", color="simulator", log_y=True).show()
