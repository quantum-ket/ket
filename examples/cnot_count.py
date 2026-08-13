# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

from ket import *

counts, depth = qulib.gate_count(lambda q: ctrl(q[:-1], X)(q[-1]), 3)

print("Gate Counts:", counts)
print("CNOT Depth:", depth)
