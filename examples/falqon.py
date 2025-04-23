# SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

from ket import Process, H, Y, Z, ham, RZZ, RX, exp_value, sample
from plotly import express as px


edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
n = 4

delta_t = 0.1

beta = [0]
cost = []

num_layers = 20

process = Process(num_qubits=n, simulator="dense", execution="live")

qubits = H(process.alloc(n))
edges = list(map(qubits.at, edges))

with ham():
    h_c = -0.5 * sum(1 - Z(i) * Z(j) for i, j in edges)
    h_b = sum(Y(i) * Z(j) + Z(i) * Y(j) for i, j in edges)

for _ in range(num_layers):
    for a, b in edges:
        RZZ(delta_t, a, b)
    RX(beta[-1] * delta_t, qubits)

    beta.append(-exp_value(h_b).get())
    cost.append(exp_value(h_c).get())

result = sample(qubits)

px.line(y=beta, title="Beta").show()
px.line(y=cost, title="Cost").show()
result.histogram().show()
