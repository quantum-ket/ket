# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Quantum phase estimation.

Estimates φ in U|1⟩ = e^{2πiφ}|1⟩, here for U = P(2πφ) with φ = π/10, so the
result times 10 approximates π.
"""

from math import pi

from ket import *


def estimate_phase(u, precision: int) -> float:
    """Return φ for the gate `u`, with `precision` bits."""
    p = Process(simulator="dense", num_qubits=precision + 1)
    c = H(p.alloc(precision))
    target = X(p.alloc())  # eigenstate |1⟩
    for i, qubit in enumerate(c):  # c[i] controls U^(2^i)
        with control(qubit):
            for _ in range(2**i):
                u(target)
    adj(QFT)(c)
    # c[0] holds the least significant bit; `measure` reads c[0] as the most.
    return measure(reversed(c)).value / 2**precision


def u(qubit):
    """P(2πφ) with φ = π/10."""
    P(2 * pi * pi / 10, qubit)


print("π ≈", 10 * estimate_phase(u, 8))
