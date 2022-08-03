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

from ..base import base_H, base_RX, base_RZ, base_X, base_Z
from ..standard import ctrl, around, base_flipc
from math import pi


def base_cnot(c, t):
    for i, j in zip(c, t):
        ctrl(i, base_X, j)


def base_swap(a, b):
    base_cnot(a, b)
    base_cnot(b, a)
    base_cnot(a, b)


def base_RXX(theta, a, b):
    for qa, qb in zip(a, b):
        with around([lambda a, b: base_H(a+b), base_cnot], qa, qb):
            base_RZ(theta, qb)


def base_RZZ(theta, a, b):
    for qa, qb in zip(a, b):
        with around(base_cnot, qa, qb):
            base_RZ(theta, qb)


def base_RYY(theta, a, b):
    for qa, qb in zip(a, b):
        with around([lambda a, b: base_RX(pi/2, a+b), base_cnot], qa, qb):
            base_RZ(theta, qb)


def base_phase_on(state, q):
    with around(lambda q: base_flipc(state, q), q):
        ctrl(q[1:], base_Z, q[0])
