from ket.gates import *
from ket.standard import ctrl, adj
from ket.lib import dump_matrix
from math import sqrt, pi, cos, sin
from cmath import exp
from ket import kbw

kbw.use_sparse()

gate_matrix = {
    I: [[1, 0],
        [0, 1]],

    X: [[0, 1],
        [1, 0]],

    Y: [[0, -1j],
        [1j,  0]],

    Z: [[1,   0],
        [0, -1]],

    H: [[1/sqrt(2),  1/sqrt(2)],
        [1/sqrt(2), -1/sqrt(2)]],

    S: [[1,   0],
        [0,  1j]],

    SD: [[1,    0],
         [0,  -1j]],

    T: [[1,             0],
        [0,  exp(1j*pi/4)]],

    TD: [[1,              0],
         [0,  exp(-1j*pi/4)]]
}

param_gate_matrix = {
    phase: lambda theta: [[1,               0],
                          [0, exp(1j*theta)]],

    RX: lambda theta: [[cos(theta/2), -1j*sin(theta/2)],
                       [-1j*sin(theta/2), cos(theta/2)]],

    RY: lambda theta: [[cos(theta/2), -sin(theta/2)],
                       [sin(theta/2),  cos(theta/2)]],

    RZ: lambda theta: [[exp(-1j*theta/2), 0],
                       [0,  exp(1j*theta/2)]]
}


def make_ctrl_gate(mat):
    return [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0]+mat[0],
            [0, 0]+mat[1]]


def make_adj_gate(mat):
    return [[complex(mat[j][i]).conjugate() for j in range(len(mat))] for i in range(len(mat[0]))]


def tau_range(parts=100):
    part = 2*pi/parts
    step = 0
    while step < 2*pi:
        yield step
        step += part
    return 2*pi


def is_equal(a, b, epsilon=1e-10):
    return all(all(abs(aj-bj) < epsilon for aj, bj in zip(ai, bi)) for ai, bi in zip(a, b))


def test_gates():
    for gate in [I, X, Y, Z, H, S, SD, T, TD]:
        for sim in [kbw.use_sparse, kbw.use_dense]:
            sim()
            print(gate, sim)
            assert is_equal(dump_matrix(gate), gate_matrix[gate])
            assert is_equal(dump_matrix(ctrl(0, gate, 1), 2),
                            make_ctrl_gate(gate_matrix[gate]))
            assert is_equal(dump_matrix(adj(gate)),
                            make_adj_gate(gate_matrix[gate]))
            assert is_equal(dump_matrix(ctrl(0, adj(gate), 1), 2),
                            make_ctrl_gate(make_adj_gate(gate_matrix[gate])))


def test_param_gates():
    for gate in [phase, RX, RY, RZ]:
        for sim in [kbw.use_sparse, kbw.use_dense]:
            sim()
            print(gate, sim)

            assert all(is_equal(dump_matrix(gate(theta)),
                                param_gate_matrix[gate](theta)) for theta in tau_range())
            assert all(is_equal(dump_matrix(ctrl(0, gate(theta), 1), 2),
                                make_ctrl_gate(param_gate_matrix[gate](theta))) for theta in tau_range())
            assert all(is_equal(dump_matrix(adj(gate(theta))),
                                make_adj_gate(param_gate_matrix[gate](theta))) for theta in tau_range())
            assert all(is_equal(dump_matrix(ctrl(0, adj(gate(theta)), 1), 2),
                                make_ctrl_gate(make_adj_gate(param_gate_matrix[gate](theta)))) for theta in tau_range())
