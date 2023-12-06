# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from ket import quant, X, Z, H, measure, ctrl, code_ket

@code_ket
def teleport(alice : quant) -> quant:
    alice_b, bob_b = quant(2)
    ctrl(H(alice_b), X, bob_b)

    ctrl(alice, X, alice_b)
    H(alice)

    m0 = measure(alice)
    m1 = measure(alice_b)

    if m1 == 1:
        X(bob_b)
    if m0 == 1:
        Z(bob_b)

    return bob_b

alice = quant(1)         # alice = |0⟩
H(alice)                 # alice = |+⟩
Z(alice)                 # alice = |–⟩
bob = teleport(alice)    # bob  <- alice
H(bob)                   # bob   = |1⟩
bob_m = measure(bob)

print('Expected measure 1, result =', bob_m.value)
# Expected measure 1, result = 1     