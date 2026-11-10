# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Deutsch algorithm.

Decides if f: {0, 1} → {0, 1} is constant or balanced with a single call to
its oracle U|x⟩|y⟩ = |x⟩|y ⊕ f(x)⟩.
"""

from ket import *


def balanced(x, y):
    """f(x) = x."""
    CNOT(x, y)


def constant(_x, y):
    """f(x) = 1."""
    X(y)


def deutsch(oracle) -> str:
    """Run the algorithm and return "constant" or "balanced"."""
    p = Process()
    x, y = p.alloc(2)
    H(x)  # |+⟩
    H(X(y))  # |–⟩
    oracle(x, y)
    return "balanced" if measure(H(x)).value else "constant"


print("balanced oracle:", deutsch(balanced))
print("constant oracle:", deutsch(constant))
