from ket.gates import *
from ket.standard import ctrl
from ket.lib import dump_matrix
from math import sqrt, pi, cos, sin
from cmath import exp

gate_list = [I, X, Y, Z, H, S, SD, T, TD]
param_gate_list = [phase, RX, RY, RZ]

gate_matrix = {
    I : [[1, 0],
         [0, 1]],

    X : [[0, 1],
         [1, 0]],

    Y : [[0, -1j],
         [1j,  0]],

    Z : [[1,   0],
         [0, -1]],

    H : [[1/sqrt(2),  1/sqrt(2)],
         [1/sqrt(2), -1/sqrt(2)]],

    S : [[1,   0],
         [0,  1j]],

    SD : [[1,    0],
          [0,  -1j]],
    
    T : [[1,             0],
         [0,  exp(1j*pi/4)]],

    TD : [[1,              0],
         [0,  exp(-1j*pi/4)]]
}

param_gate_matrix = {
    phase : lambda _lambda : [[1,               0],
                              [0, exp(1j*_lambda)]],

    RX    : lambda theta : [[cos(theta/2), -1j*sin(theta/2)],
                            [-1j*sin(theta/2), cos(theta/2)]],

    RY    : lambda theta : [[cos(theta/2), -sin(theta/2)],
                            [sin(theta/2),  cos(theta/2)]],

    RZ    : lambda theta : [[exp(-1j*theta/2), 0],
                            [0,  exp(1j*theta/2)]]                     
}

def make_ctrl_gate(mat):
    return [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0]+mat[0],
            [0, 0]+mat[1]]

def is_equal(a, b, epsilon=1e-10):
    return all(all(abs(aj-bj) < epsilon for aj, bj in zip(ai, bi)) for ai, bi in zip(a, b))

def test_gates(test):
    not_fail = True
    for gate in gate_list:
        result = test(gate)
        not_fail = not_fail and result
        print(gate, 'OK' if result else 'FAIL!', sep='\t')
    return not_fail

def test_param_gates(test, step=100):
    step_param = 2*pi/step
    not_fail = True
    for gate in param_gate_list:
        result = all(test(gate, step_param*i) for i in range(step+1))
        not_fail = not_fail and result
        print(gate, 'OK' if result else 'FAIL!', sep='\t')
    return not_fail

if __name__ == '__main__':
    print('GATES')
    done_gate = test_gates(lambda gate: is_equal(dump_matrix(gate), gate_matrix[gate])) 
    print('PARAM GATES')
    done_param_gate = test_param_gates(lambda gate, param : is_equal(dump_matrix(gate(param)), param_gate_matrix[gate](param)))
    print('CTRL GATES')
    done_ctrl_gate = test_gates(lambda gate : is_equal(dump_matrix(ctrl(0, gate, 1), 2), make_ctrl_gate(gate_matrix[gate])))
    print('PARAM CTRL GATES')
    done_param_ctrl_gate = test_param_gates(lambda gate, param : is_equal(dump_matrix(ctrl(0, gate(param), 1), 2), make_ctrl_gate(param_gate_matrix[gate](param))))
  
    if all([done_gate, done_param_gate, done_ctrl_gate, done_param_ctrl_gate]):
        print('ALL OK')

    else:
        print('FAIL!!!')
        exit(-1)