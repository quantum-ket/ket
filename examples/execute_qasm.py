# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

from ket import *

w_state = r"""
OPENQASM 2.0;
include "qelib1.inc";


qreg q[3];
creg c[3];

gate cH a,b {
    h b;
    sdg b;
    cx a,b;
    h b;
    t b;
    cx a,b;
    t b;
    h b;
    s b;
    x b;
    s a;
}

u3(1.91063,0,0) q[0];
cH q[0],q[1];
ccx q[0],q[1],q[2];
x q[0];
x q[1];
cx q[0],q[1];
"""

process = Process()
qubits = process.alloc(3)

qasm(qubits, w_state)

print(dump(qubits).show())
