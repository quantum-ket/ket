# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

from ket import *
from ket.qulib import GateCount

gc = GateCount(3)
p = Process(gc)
q = p.alloc(3)
ctrl(q[:-1], X)(q[-1])

m = measure(q)

print("Gate Counts:", gc.counts)
print("CNOT Depth:", gc.cnot_depth)
