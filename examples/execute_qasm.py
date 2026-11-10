# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Run an OpenQASM 2.0 circuit.

`qasm` applies the circuit to Ket qubits, so it can be mixed with Ket code.
This circuit prepares the W state (|001⟩ + |010⟩ + |100⟩)/√3.
"""

from ket import *

W_STATE = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
u3(1.91063, 0, 0) q[0];
ch q[0], q[1];
ccx q[0], q[1], q[2];
x q[0];
x q[1];
cx q[0], q[1];
"""

p = Process()
q = p.alloc(3)
qasm(q, W_STATE)
print(dump(q).show())
