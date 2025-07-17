"""Quantum state preparation.

Utilities for preparing quantum states.
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0


from typing import Callable, Literal
from cmath import phase
from math import sqrt, asin

from ..base import Quant
from ..gates import X, H, RY, CNOT, S, P
from ..operations import ctrl, control, around


__all__ = [
    "ghz",
    "w",
    "bell",
    "pauli_state",
    "state",
]


def bell(qubit_a: Quant, qubit_b: Quant) -> Quant:
    r"""Prepare a Bell state = :math:`\frac{1}{\sqrt{2}}(\ket{0}+\ket{1})` state."""

    return CNOT(H(qubit_a), qubit_b)


def ghz(qubits: Quant) -> Quant:
    r"""Prepare a GHZ = :math:`\frac{1}{\sqrt{2}}(\ket{0\dots0}+\ket{1\dots1})` state."""

    ctrl(H(qubits[0]), X)(qubits[1:])

    return qubits


def w(qubits: Quant) -> Quant:
    r"""Prepare a W = :math:`\frac{1}{\sqrt{n}}\sum_{k=0}^{n}\ket{2^k}` state."""

    size = len(qubits)

    X(qubits[0])
    for i in range(size - 1):
        n = size - i
        ctrl(qubits[i], RY(2 * asin(sqrt((n - 1) / n)))(qubits[i + 1]))
        CNOT(qubits[i + 1], qubits[i])

    return qubits


def pauli_state(
    pauli: Literal["X", "Y", "Z"],
    eigenvalue: Literal[+1, -1],
    qubits: Quant | None = None,
) -> Quant | Callable[[Quant], Quant]:
    """Prepare a quantum state in the +1 or -1 eigenstate of a Pauli operator.

    This function prepares a quantum state in the +1 or -1 eigenstate of a specified Pauli operator.
    The resulting quantum state can be obtained by either directly calling the function with qubits,
    or by returning a closure that can be applied to qubits later.

    Args:
        pauli: Pauli operator to prepare the eigenstate for.
        eigenvalue: Eigenvalue of the Pauli operator (+1 or -1).
        qubits: Qubits to prepare the eigenstate on. If None, returns a closure.

    Returns:
        If qubits is provided, returns the resulting quantum state.
        If qubits is None, returns a closure that can be applied to qubits later.
    """

    def inner(qubits: Quant) -> Quant:
        if eigenvalue == +1:
            pass
        elif eigenvalue == -1:
            X(qubits)
        else:
            raise ValueError("Invalid eigenvalue")

        if pauli == "X":
            return H(qubits)
        if pauli == "Y":
            return S(H(qubits))
        if pauli == "Z":
            return qubits
        raise ValueError("Invalid Pauli operator")

    if qubits is None:
        return inner
    return inner(qubits)


class _ParamTree:  # pylint: disable=too-few-public-methods
    def __init__(self, prob: list[float], amp: list[float]):
        total = sum(prob)
        prob = [p / total for p in prob]
        l_prob = prob[: len(prob) // 2]
        l_amp = amp[: len(prob) // 2]
        r_prob = prob[len(prob) // 2 :]
        r_amp = amp[len(prob) // 2 :]

        self.value = sum(r_prob)
        self.value = 2 * asin(sqrt(self.value))
        if len(prob) > 2:
            self.left = _ParamTree(l_prob, l_amp) if sum(l_prob) > 1e-10 else None
            self.right = _ParamTree(r_prob, r_amp) if sum(r_prob) > 1e-10 else None
        else:
            self.left = None
            self.right = None
            self.phase0 = amp[0]
            self.phase1 = amp[1]

    def _is_leaf(self):
        return self.left is None and self.right is None


def state(
    amp: list[complex],
    qubits: Quant,
):
    r"""Prepare a quantum state from a list of amplitudes probabilities.

    If the qubits are in the state :math:`\ket{0\dots0}`, the resulting state will be
    :math:`\sum_{i=0}^{n} a_i \ket{i}`, where :math:`a_i` are the amplitudes
    corresponding to the computational basis states.

    The amplitude probabilities are normalized automatically, so the sum of the squares
    of the absolute values of the amplitudes will be equal to 1.

    .. warning::
        The execution time of this function is exponential in the number of qubits.

    Args:
        amp: List of complex numbers representing the amplitudes of the quantum state.
        qubits: Qubits to prepare the quantum state on.
    """

    if not isinstance(amp, _ParamTree):
        if 1 << len(qubits) != len(amp):
            raise ValueError(
                f"Number of amplitudes {len(amp)} does not match the number "
                f"of qubits {len(qubits)}. For {len(qubits)} qubits, expected "
                f"{1 << len(qubits)} amplitudes."
            )
        amp = _ParamTree(list(abs(p) ** 2 for p in amp), list(phase(p) for p in amp))

    head, *tail = qubits
    RY(amp.value, head)
    if amp._is_leaf():  # pylint: disable=protected-access
        with around(X, head):
            P(amp.phase0, head)
        P(amp.phase1, head)
        return
    if amp.left is not None:
        with around(X, head):
            with control(head):
                state(amp.left, tail)
    if amp.right is not None:
        with control(head):
            state(amp.right, tail)
