# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Bell state and CHSH on Amazon Braket.

Same program as bell.py: only the Process changes. Without arguments,
AmazonBraket runs on the local Braket simulator. Pass a device ARN to use a
managed simulator or a QPU, for example
AmazonBraket("arn:aws:braket:::device/quantum-simulator/amazon/sv1").

Requires: pip install ket-lang[amazon]
"""

from math import sqrt

from ket import *
from ket.amazon import AmazonBraket

p = Process(AmazonBraket())
a, b = p.alloc(2)
CNOT(H(a), b)

with obs():
    a0, a1 = Z(a), X(a)
    b0, b1 = (Z(b) + X(b)) / sqrt(2), (Z(b) - X(b)) / sqrt(2)
    chsh = a0 * b0 + a0 * b1 + a1 * b0 - a1 * b1

print("CHSH:", exp_value(chsh).get())
