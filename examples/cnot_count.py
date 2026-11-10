# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Count the gates of a circuit after decomposition.

`qulib.gate_count` decomposes multi-controlled gates into CNOTs and
single-qubit gates, as needed to run on a QPU.
"""

from ket import *


def toffoli(q):
    """X on the last qubit, controlled by all others."""
    with control(q[:-1]):
        X(q[-1])


counts, depth = qulib.gate_count(toffoli, num_qubits=3)
print("gate counts:", dict(counts))
print("CNOT depth:", depth)
