# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Bell state and CHSH on IBM Quantum.

Same program as bell.py: only the Process changes. Without arguments,
IBMDevice runs on the local Aer simulator.

Requires: pip install ket-lang[ibm]
"""

from math import sqrt

from ket import *
from ket.ibm import IBMDevice

p = Process(IBMDevice())
a, b = p.alloc(2)
CNOT(H(a), b)

with obs():
    a0, a1 = Z(a), X(a)
    b0, b1 = (Z(b) + X(b)) / sqrt(2), (Z(b) - X(b)) / sqrt(2)
    chsh = a0 * b0 + a0 * b1 + a1 * b0 - a1 * b1

print("CHSH:", exp_value(chsh).get())
