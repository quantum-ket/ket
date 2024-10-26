# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Bell state preparation."""
import ket
from math import sqrt

import ket.lib

process = ket.Process()
a, b = process.alloc(2)

ket.X(a + b)
ket.CNOT(ket.H(a), b)

a0b0 = ket.Pauli.z(a) * -((ket.Pauli.x(b) + ket.Pauli.z(b)) / sqrt(2))
a0b1 = ket.Pauli.z(a) * ((ket.Pauli.x(b) - ket.Pauli.z(b)) / sqrt(2))
a1b0 = ket.Pauli.x(a) * -((ket.Pauli.x(b) + ket.Pauli.z(b)) / sqrt(2))
a1b1 = ket.Pauli.x(a) * ((ket.Pauli.x(b) - ket.Pauli.z(b)) / sqrt(2))

print(ket.exp_value(a0b0 + a0b1 + a1b0 - a1b1).get())
