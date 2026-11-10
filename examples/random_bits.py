# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Quantum random number generator.

A qubit in |+⟩ gives 0 or 1 with equal probability when measured.
"""

from ket import *


def random_number(n_bits: int) -> int:
    """Return a random integer in [0, 2**n_bits)."""
    p = Process(simulator="dense", num_qubits=n_bits)
    return measure(H(p.alloc(n_bits))).value


print("16-bit random number:", random_number(16))
