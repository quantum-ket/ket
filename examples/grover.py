# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Grover's search.

Finds the marked item among 2**n in about (π/4)√(2**n) oracle calls.
"""

from math import pi, sqrt
from random import randrange

from ket import *


def grover(n: int, oracle) -> int:
    """Return the state marked by `oracle` on `n` qubits."""
    p = Process(simulator="dense", num_qubits=n)
    q = H(p.alloc(n))
    for _ in range(int(pi / 4 * sqrt(2**n))):
        oracle(q)
        with around(cat(H, X), q):  # diffusion operator
            CZ(*q)
    return measure(q).value


n = 12
target = randrange(2**n)
print("looking for", target, "found", grover(n, qulib.oracle.phase_oracle(target)))
