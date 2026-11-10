# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""QAOA for Max-Cut.

Splits the nodes of a graph into two groups to cut as many edges as possible.
A classical optimizer (SciPy) tunes the angles of a QAOA circuit to maximize
the expected cut.

Requires: pip install scipy
"""

from scipy.optimize import minimize

from ket import *
from ket.qulib import ham

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # a square: the best cut is 4
NODES = 4
LAYERS = 2


def qaoa(params):
    """Prepare the QAOA state; return its qubits and the Max-Cut Hamiltonian."""
    gammas, betas = params[:LAYERS], params[LAYERS:]
    p = Process(simulator="dense", num_qubits=NODES)
    q = H(p.alloc(NODES))
    for gamma, beta in zip(gammas, betas):
        for i, j in EDGES:
            RZZ(gamma, q[i], q[j])
        RX(2 * beta, q)
    return q, ham.maxcut(EDGES, q)


def expected_cut(params) -> float:
    """The expected number of cut edges."""
    _, cut = qaoa(params)
    return exp_value(cut).get()


result = minimize(lambda x: -expected_cut(x), [0.5] * (2 * LAYERS), method="COBYLA")
print("expected cut:", -result.fun)

q, _ = qaoa(result.x)
best = sample(q).most_frequent_state()
print(f"best partition: {best:0{NODES}b}")
