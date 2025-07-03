# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Bell state preparation."""
from pprint import pprint
from ket import *
from ket.ibm import IBMDevice
from math import sqrt

from qiskit_aer import AerSimulator

sim = AerSimulator()

device = IBMDevice(sim)

process = Process(device)
a, b = process.alloc(2)

X(a + b)
CNOT(H(a), b)

with ham():
    a0 = Z(a)
    a1 = X(a)
    b0 = -(X(b) + Z(b)) / sqrt(2)
    b1 = (X(b) - Z(b)) / sqrt(2)
    h = a0 * b0 + a0 * b1 + a1 * b0 - a1 * b1

print(exp_value(h).get())
pprint(process.get_instructions())
