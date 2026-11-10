# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Quantum teleportation.

Alice sends the state of a qubit to Bob using a shared Bell pair and two
classical bits. Live execution (the default) lets the program use
measurement results right away, to choose Bob's corrections.
"""

from ket import *


def teleport(message, alice, bob):
    """Move the state of `message` to `bob`; `alice` and `bob` are a Bell pair."""
    CNOT(message, alice)
    H(message)
    if measure(alice).value:
        X(bob)
    if measure(message).value:
        Z(bob)
    return bob


p = Process()
message = Z(H(p.alloc()))  # |–⟩
alice, bob = p.alloc(2)
CNOT(H(alice), bob)  # Bell pair

bob = teleport(message, alice, bob)
print("Bob's state:")
print(dump(bob).show())  # |–⟩ = (|0⟩ − |1⟩)/√2
