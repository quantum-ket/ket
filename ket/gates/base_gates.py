#  Copyright 2020, 2021 Evandro Chagas Ribeiro da Rosa <evandro.crr@posgrad.ufsc.br>
#  Copyright 2020, 2021 Rafael de Santiago <r.santiago@ufsc.br>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

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
