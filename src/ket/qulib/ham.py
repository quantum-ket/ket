"""Hamiltonian library."""

# SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from ..gates import X, Y, Z, B, obs
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


def knapsack(
    weights: list[float],
    values: list[float],
    capacity: float,
    qubits: Quant,
    penalty: None | float = None,
) -> Hamiltonian:
    r"""Knapsack Hamiltonian.

    Implements the QUBO formulation of the 0/1 Knapsack problem
    using a quadratic penalty to enforce the capacity constraint.

    The Hamiltonian minimized is:

    .. math::

        -\sum_i v_i x_i
        +
        A \left(\sum_i w_i x_i - C\right)^2

    where :math:`x_i = \frac{1 - Z_i}{2}` and :math:`A` is the
    penalty coefficient.

    If ``penalty`` is not provided, it defaults to
    ``max(values)``, which is typically sufficient to discourage
    capacity violations in practical QAOA runs.

    Args:
        weights: List of item weights :math:`w_i`.
        values: List of item values :math:`v_i`.
        capacity: Maximum allowed total weight :math:`C`.
        qubits: Qubits representing the binary decision variables.
        penalty: Penalty coefficient :math:`A`. If ``None``,
            defaults to ``max(values)``.

    Returns:
        Hamiltonian representing the knapsack objective.
    """
    if penalty is None:
        penalty = max(values)

    with obs():
        objective = sum(v * B(q) for v, q in zip(values, qubits))
        restriction = (sum(w * B(q) for w, q in zip(weights, qubits)) - capacity) ** 2
        return -objective + penalty * restriction


def tsp(
    cities: dict[tuple[int, int], float],
    n: int,
    qubits: list,
    penalty: float | None = None,
) -> Hamiltonian:
    r"""Traveling Salesman Problem (TSP) Hamiltonian.

    Implements the QUBO formulation of the TSP using a
    quadratic penalty to enforce permutation constraints.

    The Hamiltonian minimized is:

    .. math::

        \sum_{t=0}^{n-1} \sum_{i,j} d_{ij} x_{i,t} x_{j,t+1}
        +
        A \sum_i \left(\sum_t x_{i,t} - 1\right)^2
        +
        A \sum_t \left(\sum_i x_{i,t} - 1\right)^2

    where:

    - :math:`d_{ij}` is the directed distance from city :math:`i`
      to city :math:`j`,
    - :math:`x_{i,t} \in \{0,1\}` indicates whether city :math:`i`
      is visited at tour position :math:`t`,
    - :math:`x_{i,t} = \frac{1 - Z_{i,t}}{2}`,
    - :math:`A` is the penalty coefficient.

    The first term minimizes the total tour length.
    The second term enforces that each city is visited exactly once.
    The third term enforces that exactly one city is assigned to
    each tour position.

    If ``penalty`` is not provided, it defaults to
    ``2 * max(cities.values())``, which is typically sufficient
    to discourage constraint violations.

    Args:
        cities: Dictionary mapping directed edges ``(i, j)`` to
            distances :math:`d_{ij}`.
        n: Number of cities.
        qubits: Flat list of :math:`n^2` qubits representing the
            binary variables :math:`x_{i,t}`.
        penalty: Penalty coefficient :math:`A`. If ``None``,
            defaults to ``2 * max(cities.values())``.

    Returns:
        Hamiltonian representing the directed TSP cost function
        with quadratic constraint penalties.
    """
    if penalty is None:
        penalty = 2 * max(cities.values())

    # Reshape flat qubit list into n x n grid
    qubits = [qubits[i * n : (i + 1) * n] for i in range(n)]

    visit_once = sum(
        (sum(B(qubits[i][t]) for t in range(n)) - 1) ** 2 for i in range(n)
    )

    one_per_time = sum(
        (sum(B(qubits[i][t]) for i in range(n)) - 1) ** 2 for t in range(n)
    )

    distance = sum(
        sum(
            d * B(qubits[i][t]) * B(qubits[j][(t + 1) % n])
            for (i, j), d in cities.items()
        )
        for t in range(n)
    )

    return distance + penalty * (visit_once + one_per_time)


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
        return sum(X(a) * X(b) + Y(a) * Y(b) for a, b in map(qubits.at, edges)) / 2


def qubo(model, qubits: Quant) -> Hamiltonian:
    """Convert a QUBO model to a Hamiltonian.

    Converts a QUBO model from the pyQUBO library to a Hamiltonian observable.

    Args:
        model: QUBO model.
        qubits: Qubits representing the variables in the model.
    """
    warnings.warn(
        "`qulib.ham.qubo` is deprecated and will be removed in future versions."
        " Use `B(x)` instead to construct the Hamiltonian with binary values.",
        DeprecationWarning,
        stacklevel=2,
    )

    linear, quadratic, offset = model.to_ising(index_label=True)

    with obs():
        return (
            offset
            + sum(c * Z(qubits[i]) for i, c in linear.items())
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
    return 1j * commutator(hp, hd)


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
