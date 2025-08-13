"""Quantum oracles."""

from __future__ import annotations

# SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

from functools import wraps
from typing import Callable

from ..base import Quant
from ..operations import control, ctrl, around
from ..gates import X, Z
from . import math


__all__ = [
    "xor_oracle",
    "phase_oracle",
]


def xor_oracle(func):
    r"""Create an inefficient xor oracle for a given function.

    This oracle will compute the xor of the function's output with the input.
    The function must take an integer input and return an integer output.

    .. math::
        \left|x\right>\left|y\right> \mapsto \left|x\right>\left|y \oplus f(x)\right>

    Example:

    .. code-block:: python

        def my_function(x: int) -> int:
            return x + 1
        oracle = xor_oracle(my_function)
        oracle(x, y)

    Args:
        func: Function to create the oracle for.

    Returns:
        Oracle that computes the xor of the function's output with the input.
    """

    @wraps(func)
    def inner(x: Quant, y: Quant):
        for state in range(2 ** len(x)):
            with control(x, state):
                math.set_int(y, func(state))

    return inner


def phase_oracle(
    state: int, qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:
    r"""Transform qubits from :math:`\ket{\texttt{state}}` to :math:`-\ket{\texttt{state}}`.

    This gate is useful for marking states in Grover's algorithm.
    """

    def inner(qubits: Quant) -> Quant:
        init, last = qubits[:-1], qubits[-1]
        with around(lambda q: X(q) if state & 1 == 0 else None, last):
            ctrl(init, Z, state >> 1)(last)
        return qubits

    if qubits is None:
        return inner
    return inner(qubits)
