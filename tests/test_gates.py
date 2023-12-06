# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from ket.gates import *
from ket.base import set_process_features
from ket.standard import ctrl, adj
from ket.lib import dump_matrix
from math import sqrt, pi, cos, sin
from cmath import exp
from ket import kbw
import numpy as np

kbw.use_sparse()

GATE_MATRIX = {
    I: [[1, 0],
        [0, 1]],

    X: [[0, 1],
        [1, 0]],

    Y: [[0, -1j],
        [1j, 0]],

    Z: [[1, 0],
        [0, -1]],

    H: [[1 / sqrt(2), 1 / sqrt(2)],
        [1 / sqrt(2), -1 / sqrt(2)]],

    S: [[1, 0],
        [0, 1j]],

    SD: [[1, 0],
         [0, -1j]],

    T: [[1, 0],
        [0, exp(1j * pi / 4)]],

    TD: [[1, 0],
         [0, exp(-1j * pi / 4)]]
}

PARAM_GATE_MATRIX = {
    phase: lambda theta: [[1, 0],
                          [0, exp(1j * theta)]],

    RX: lambda theta: [[cos(theta / 2), -1j * sin(theta / 2)],
                       [-1j * sin(theta / 2), cos(theta / 2)]],

    RY: lambda theta: [[cos(theta / 2), -sin(theta / 2)],
                       [sin(theta / 2), cos(theta / 2)]],

    RZ: lambda theta: [[exp(-1j * theta / 2), 0],
                       [0, exp(1j * theta / 2)]]
}


def make_ctrl_gate(mat):
    return [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0] + mat[0],
            [0, 0] + mat[1]]


def make_ctrl_2_gate(mat):
    return [[1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0] + mat[0],
            [0, 0, 0, 0, 0, 0] + mat[1]]


def make_adj_gate(mat):
    return [[complex(mat[j][i]).conjugate() for j in range(len(mat))] for i in range(len(mat[0]))]


def tau_range(parts=100):
    part = 2 * pi / parts
    step = 0
    while step < 2 * pi:
        yield step
        step += part
    return 2 * pi


def is_equal(a, b, epsilon=1e-10):
    a = np.array(a, dtype=complex)
    b = np.array(b, dtype=complex)

    abdg = a @ b.conjugate().T
    abdg *= abdg[0, 0].conjugate()

    eye = np.eye(a.shape[0], dtype=complex)

    return np.isclose(abdg, eye, atol=epsilon).all()


def test_gates():
    for gate in [I, X, Y, Z, H, S, SD, T, TD]:
        for sim in [kbw.use_sparse, kbw.use_dense]:
            sim()
            print(gate, sim)
            assert is_equal(dump_matrix(gate), GATE_MATRIX[gate])
            assert is_equal(dump_matrix(ctrl(0, gate, 1), 2),
                            make_ctrl_gate(GATE_MATRIX[gate]))
            assert is_equal(dump_matrix(adj(gate)),
                            make_adj_gate(GATE_MATRIX[gate]))
            assert is_equal(dump_matrix(ctrl(0, adj(gate), 1), 2),
                            make_ctrl_gate(make_adj_gate(GATE_MATRIX[gate])))
            assert is_equal(dump_matrix(ctrl([0, 1], gate, 2), 3),
                            make_ctrl_2_gate(GATE_MATRIX[gate]))
            assert is_equal(dump_matrix(ctrl([0, 1], adj(gate), 2), 3),
                            make_ctrl_2_gate(make_adj_gate(GATE_MATRIX[gate])))


def test_param_gates():
    for gate in [phase, RX, RY, RZ]:
        for sim in [kbw.use_sparse, kbw.use_dense]:
            sim()
            print(gate, sim)

            assert all(is_equal(dump_matrix(gate(theta)),
                                PARAM_GATE_MATRIX[gate](theta)) for theta in tau_range())
            assert all(is_equal(dump_matrix(ctrl(0, gate(theta), 1), 2),
                                make_ctrl_gate(PARAM_GATE_MATRIX[gate](theta))) for theta in tau_range())
            assert all(is_equal(dump_matrix(adj(gate(theta))),
                                make_adj_gate(PARAM_GATE_MATRIX[gate](theta))) for theta in tau_range())
            assert all(is_equal(dump_matrix(ctrl(0, adj(gate(theta)), 1), 2),
                                make_ctrl_gate(make_adj_gate(PARAM_GATE_MATRIX[gate](theta)))) for theta in tau_range())
            assert all(is_equal(dump_matrix(ctrl([0, 1], gate(theta), 2), 3),
                                make_ctrl_2_gate(PARAM_GATE_MATRIX[gate](theta))) for theta in tau_range())
            assert all(is_equal(dump_matrix(ctrl([0, 1], adj(gate(theta)), 2), 3),
                                make_ctrl_2_gate(make_adj_gate(PARAM_GATE_MATRIX[gate](theta)))) for theta in tau_range())


def test_gates_decompose():
    set_process_features(decompose=True)
    test_param_gates()


def test_param_gates_decompose():
    set_process_features(decompose=True)
    test_param_gates()


def test_gates_rz():
    set_process_features(decompose=True, use_rz_as_phase=True)
    test_param_gates()


def test_param_gates_rz():
    set_process_features(decompose=True, use_rz_as_phase=True)
    test_param_gates()
