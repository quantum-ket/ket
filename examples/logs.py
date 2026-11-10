# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Logging of Ket, Libket and KBW.

Ket logs through Python's standard :mod:`logging` module. Records from the
Rust runtime (Libket) and the simulator (KBW) go to the ``ket.libket.*`` and
``ket.kbw.*`` loggers. Run with ``KET_LOG=debug`` instead of calling
``ket.set_log_level`` to get the same output without changing the code.
"""

import logging
import sys

import ket
from ket import *

logging.basicConfig(stream=sys.stdout, format="[%(levelname)-7s %(name)s] %(message)s")

# Debug shows the process configuration, each execution, and how
# multi-controlled gates are decomposed into CNOTs.
ket.set_log_level("debug")

print("== Multi-controlled gates with decomposition ==")
process = Process(num_qubits=12, simulator="dense", decompose=True)
q = process.alloc(8)
H(q[:5])
with control(q[:5]):  # the 5 controls are shared through an auxiliary qubit
    X(q[6])
    H(q[7])
print("result:", measure(q).get())

# Info only shows the main events, such as the shared controls above.
ket.set_log_level("info")

print("\n== Errors: a short message, with the cause in the log ==")
try:
    Process(num_qubits=40, simulator="dense")
except Exception as error:  # pylint: disable=broad-exception-caught
    print("exception:", error)

process = Process(num_qubits=2)
a, b = process.alloc(2)
try:
    qasm(a + b, 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0]\ncx q[0], q[1];')
except Exception as error:  # pylint: disable=broad-exception-caught
    print("exception:", error)
