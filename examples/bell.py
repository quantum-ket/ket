# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Bell state and the CHSH inequality.

Prepares an entangled pair and measures the CHSH observable. Quantum
mechanics reaches 2√2 ≈ 2.83, above the classical limit of 2.
"""

from math import sqrt

from ket import *

p = Process()
a, b = p.alloc(2)
CNOT(H(a), b)  # (|00⟩ + |11⟩)/√2

print("samples:", sample(a + b).get())  # only 0 (|00⟩) and 3 (|11⟩)

with obs():
    a0, a1 = Z(a), X(a)
    b0, b1 = (Z(b) + X(b)) / sqrt(2), (Z(b) - X(b)) / sqrt(2)
    chsh = a0 * b0 + a0 * b1 + a1 * b0 - a1 * b1

print("CHSH:", exp_value(chsh).get())  # 2.828...
