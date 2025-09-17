"""Hamiltonian library."""

# SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

from ..gates import X, Y, Z, obs
from ..base import Quant
from ..expv import Hamiltonian, commutator
from ..operations import exp_value

__all__ = [
    "maxcut",
    "x_mixer",
    "xy_mixer",
    "qubo",
    "falqon_a",
    "falqon_b",
    "falqon_a",
    "falqon_get_beta_fo",
    "falqon_get_beta_so",
]


def maxcut(edges: list[tuple[int, int]], qubits: Quant) -> Hamiltonian:
    r"""Max-Cut Hamiltonian.

    .. math::
        \sum_{a, b\,\in\, \mathcal{E}}\frac{1}{2}(1-Z_a Z_b)

    Args:
        edges: List of edges in the graph.
        qubits: Qubits representing the graph nodes.
    """
    with obs():
        return 1 / 2 * sum(1 - Z(a) * Z(b) for a, b in map(qubits.at, edges))


def x_mixer(qubits: Quant) -> Hamiltonian:
    r"""X-Mixer Hamiltonian.

    .. math::
        \sum_j X_j

    Args:
        qubits: Qubits to apply the mixer.
    """
    with obs():
        return sum(X(q) for q in qubits)


def xy_mixer(qubits, edges: list[tuple[int, int]] | None = None) -> Hamiltonian:
    r"""XY-Mixer Hamiltonian.

    .. math::
        \frac{1}{2}\sum_{a, b\,\in\, \mathcal{E}} X_a X_b + Y_a Y_b


    If edges is None, a ring topology is assumed.

    Args:
        qubits: Qubits to apply the mixer.
        edges: List of edges in the graph.
    """
    if edges is None:
        n = len(qubits)
        edges = [(i, (i + 1) % n) for i in range(n)]

    with obs():
        return (
            sum(X(a) * X(b) for a, b in map(qubits.at, edges))
            + sum(Y(a) * Y(b) for a, b in map(qubits.at, edges))
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


def falqon_a(hp: Hamiltonian, hd: Hamiltonian) -> Hamiltonian:
    """FALQON A operator.

    .. math::
        A = i[H_d, H_p]


    See https://arxiv.org/abs/2103.08619.

    Args:
        hp: Problem Hamiltonian.
        hd: Driver Hamiltonian.
    """
    return 1j * commutator(hd, hp)


def falqon_b(hp: Hamiltonian, hd: Hamiltonian) -> Hamiltonian:
    """FALQON B operator.

    .. math::
        B = [ [H_d, H_p], H_d ]


    See https://arxiv.org/abs/2407.17810.

    Args:
        hp: Problem Hamiltonian.
        hd: Driver Hamiltonian.
    """
    return commutator(commutator(hd, hp), hd)


def falqon_c(hp: Hamiltonian, hd: Hamiltonian) -> Hamiltonian:
    """FALQON C operator.

    .. math::
        C = [ [H_d, H_p], H_p ]

    See https://arxiv.org/abs/2407.17810.

    Args:
        hp: Problem Hamiltonian.
        hd: Driver Hamiltonian.
    """
    return commutator(commutator(hd, hp), hp)


def falqon_get_beta_fo(hp: Hamiltonian, hd: Hamiltonian) -> float:
    """Get FALQON first-order beta parameter.

    See https://arxiv.org/abs/2103.08619.

    This function computes the expectation value, triggering the circuit execution.

    Args:
        hp: Problem Hamiltonian.
        hd: Driver Hamiltonian.
    """
    return -exp_value(falqon_a(hp, hd)).get()


def falqon_get_beta_so(delta_t, hp: Hamiltonian, hd: Hamiltonian) -> float:
    """Get FALQON second-order beta parameter.

    See https://arxiv.org/abs/2407.17810.

    This function computes the expectation values, triggering the circuit execution.

    Args:
        delta_t: Time step.
        hp: Problem Hamiltonian.
        hd: Driver Hamiltonian.
    """
    a = exp_value(falqon_a(hp, hd))
    b = exp_value(falqon_b(hp, hd))
    c = exp_value(falqon_c(hp, hd))

    a = a.get()
    b = b.get()
    c = c.get()

    so = -(a + delta_t * c) / (2 * delta_t * b)

    return -a if abs(so) > abs(a) else so
