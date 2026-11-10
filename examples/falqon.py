# SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

"""FALQON: feedback-based quantum optimization for Max-Cut.

Each layer measures ⟨H_b⟩ and uses it to set the next angle, so no classical
optimizer is needed. The cost goes down layer by layer.

Requires: pip install ket-lang[plot]
"""

from plotly import express as px

from ket import *

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
NODES = 4
LAYERS = 20
DELTA_T = 0.1

p = Process(simulator="dense", num_qubits=NODES)
q = H(p.alloc(NODES))

with obs():
    h_c = -0.5 * sum(1 - Z(q[i]) * Z(q[j]) for i, j in EDGES)  # cost
    h_b = sum(Y(q[i]) * Z(q[j]) + Z(q[i]) * Y(q[j]) for i, j in EDGES)

beta, cost = [0.0], []
for _ in range(LAYERS):
    for i, j in EDGES:
        RZZ(DELTA_T, q[i], q[j])
    RX(beta[-1] * DELTA_T, q)
    beta.append(-exp_value(h_b).get())
    cost.append(exp_value(h_c).get())

print("cost per layer:", [round(c, 3) for c in cost])
px.line(y=cost, title="FALQON cost", labels={"x": "layer", "y": "⟨H_c⟩"}).show()
sample(q).histogram().show()
