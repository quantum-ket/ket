# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from math import pi
from ..base import base_H, base_RX, base_RZ, base_X, base_Z
from ..standard import ctrl, around
from ..standard.ctrl import base_flipc

# pylint: disable=missing-function-docstring, invalid-name


def base_cnot(ctrl_qubits, trg_qubits):
    for ctrl_qubit, trg_qubit in zip(ctrl_qubits, trg_qubits):
        ctrl(ctrl_qubit, base_X, trg_qubit)


def base_swap(qubits_a, qubits_b):
    base_cnot(qubits_a, qubits_b)
    base_cnot(qubits_b, qubits_a)
    base_cnot(qubits_a, qubits_b)


def base_RXX(theta, qubits_a, qubits_b):
    for qubit_a, qubit_b in zip(qubits_a, qubits_b):
        with around([lambda a, b: base_H(a + b), base_cnot], qubit_a, qubit_b):
            base_RZ(theta, qubit_b)


def base_RZZ(theta, qubits_a, qubits_b):
    for qubit_a, qubit_b in zip(qubits_a, qubits_b):
        with around(base_cnot, qubit_a, qubit_b):
            base_RZ(theta, qubit_b)


def base_RYY(theta, qubits_a, qubits_b):
    for qubit_a, qubit_b in zip(qubits_a, qubits_b):
        with around([lambda a, b: base_RX(pi / 2, a + b), base_cnot], qubit_a, qubit_b):
            base_RZ(theta, qubit_b)


def base_phase_on(state, qubits):
    with around(lambda q: base_flipc(state, q), qubits):
        ctrl(qubits[1:], base_Z, qubits[0])
