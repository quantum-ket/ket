# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

from ket import *
from ket.qulib import GateCount, BatchGateCount, gate_count


def test_gatecount():
    gc = GateCount(3)
    p = Process(gc)
    q = p.alloc(3)
    H(q[0])
    CNOT(q[0], q[1])
    X(q[2])
    CNOT(q[1], q[2])
    H(q)
    measure(q)

    assert gc.counts["h"] == 4
    assert gc.counts["cnot"] == 2
    assert gc.counts["x"] == 1
    assert gc.cnot_depth == 2


def test_batch_gatecount():
    bgc = BatchGateCount(3)
    p = Process(bgc)
    q = p.alloc(3)
    H(q[0])
    CNOT(q[0], q[1])
    X(q[2])
    CNOT(q[1], q[2])
    H(q)

    # Must evaluate process for batch mode
    with obs():
        exp_value(Z(q[0])).get()

    assert bgc.counts["h"] == 4
    assert bgc.counts["cnot"] == 2
    assert bgc.counts["x"] == 1
    assert bgc.cnot_depth == 2


def test_gate_count_func_batch():
    def my_gate(q):
        H(q[0])
        CNOT(q[0], q[1])
        X(q[2])
        CNOT(q[1], q[2])
        H(q)

    counts, depth = gate_count(my_gate, 3, batch=True)
    assert counts["h"] == 4
    assert counts["cnot"] == 2
    assert counts["x"] == 1
    assert depth == 2
