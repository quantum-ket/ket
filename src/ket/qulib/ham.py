"""Hamiltonian Library."""

# SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

from ..gates import X, Y, Z, obs
from ..base import Quant
from ..expv import Hamiltonian

__all__ = [
    "maxcut",
]


def maxcut(vertices: list[tuple[int, int]], qubits: Quant) -> Hamiltonian:
    r"""Max-Cut Hamiltonian.

    .. math::
        \sum_{a, b\,\in\, \mathcal{V}}\frac{1}{2}(1-Z_a Z_b)

    Args:
        vertices: List of edges in the graph.
        qubits: Qubits representing the graph nodes.
    """
    with obs():
        return 1 / 2 * sum(1 - Z(a) * Z(b) for a, b in map(qubits.at, vertices))


def x_mixer(qubits: Quant) -> Hamiltonian:
    r"""X-Mixer Hamiltonian.

    .. math::
        \sum_j X_j

    Args:
        qubits: Qubits to apply the mixer.
    """
    with obs():
        return sum(X(q) for q in qubits)


def xy_mixer(qubits, vertices: list[tuple[int, int]] | None = None) -> Hamiltonian:
    r"""XY-Mixer Hamiltonian.

    .. math::
        \frac{1}{2}\sum_{a, b\,\in\, \mathcal{V}} X_a X_b + Y_a Y_b


    If vertices is None, a ring topology is assumed.

    Args:
        qubits: Qubits to apply the mixer.
        vertices: List of edges in the graph.
    """
    if vertices is None:
        n = len(qubits)
        vertices = [(i, (i + 1) % n) for i in range(n)]

    with obs():
        return (
            sum(X(a) * X(b) for a, b in map(qubits.at, vertices))
            + sum(Y(a) * Y(b) for a, b in map(qubits.at, vertices))
        ) / 2


def qubo(model, qubits: Quant) -> Hamiltonian:
    """Convert a QUBO model to a Hamiltonian.

    Converts a QUBO model from the pyQUBO library to a Hamiltonian observable.

    Args:
        model: QUBO model.
        qubits: Qubits representing the variables in the model.
    """
    linear, quadratic, offset = model.to_ising(index_label=True)

    with obs():
        return (
            offset
            + sum(c * -Z(qubits[i]) for i, c in linear.items())
            + sum(c * Z(qubits[i]) * Z(qubits[j]) for (i, j), c in quadratic.items())
        )
