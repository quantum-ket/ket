# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import pytest

from ket import Process, B, obs, exp_value
from ket.qulib import exact_solver, energy
from ket.qulib.qubo import (
    QUBO,
    Expression,
    Constraint,
    binary,
    one_hot,
    slack_size,
    exactly_one,
    at_most_one,
    equal,
    less_equal,
    greater_equal,
)
from ket.qulib.qubo import _evaluate


def all_bits(n):
    return [list(bits) for bits in product((0, 1), repeat=n)]


# --- Encodings ---


@pytest.mark.parametrize("max_value", range(1, 20))
def test_binary_covers_exactly_zero_to_max(max_value):
    n = slack_size(max_value, "binary")
    with obs(n) as q:
        v = binary(q, max_value)
    values = {round(_evaluate(v, bits)) for bits in all_bits(n)}
    assert values == set(range(max_value + 1))


def test_binary_weights_max_5():
    # Weights 1, 2 and 2: the last qubit completes the maximum.
    with obs(3) as q:
        v = binary(q, 5)
    assert [_evaluate(v, b) for b in ([1, 0, 0], [0, 1, 0], [0, 0, 1])] == [1, 2, 2]


def test_binary_default_is_plain_binary():
    with obs(3) as q:
        v = binary(q)
    assert sorted(round(_evaluate(v, b)) for b in all_bits(3)) == list(range(8))


def test_binary_wrong_size():
    with obs(4) as q:
        with pytest.raises(ValueError):
            binary(q, 5)


def test_slack_size():
    assert slack_size(5) == 3
    assert slack_size(8) == 4
    assert slack_size(5, "one_hot") == 6
    with pytest.raises(ValueError):
        slack_size(3, "unary")


def test_one_hot_value():
    with obs(3) as q:
        v = one_hot(q, [10, 20, 30])
    assert _evaluate(v, [0, 1, 0]) == 20


# --- Restrictions ---


def test_exactly_one():
    with obs(3) as q:
        h = exactly_one(q)
    for bits in all_bits(3):
        assert (_evaluate(h, bits) == 0) == (sum(bits) == 1)


def test_at_most_one():
    with obs(3) as q:
        h = at_most_one(q)
    for bits in all_bits(3):
        assert (abs(_evaluate(h, bits)) < 1e-9) == (sum(bits) <= 1)


def test_equal():
    with obs(3) as q:
        h = equal(B(q[0]) + B(q[1]) + B(q[2]), 2)
    for bits in all_bits(3):
        assert (abs(_evaluate(h, bits)) < 1e-9) == (sum(bits) == 2)


@pytest.mark.parametrize("encoding", ["binary", "one_hot"])
def test_less_equal(encoding):
    # 2*x0 + 3*x1 <= 3 ; slack must reach 3
    m = slack_size(3, encoding)
    with obs(2 + m) as q:
        expr = 2 * B(q[0]) + 3 * B(q[1])
        h = less_equal(expr, 3, q[2:], encoding)
    for x in all_bits(2):
        best = min(_evaluate(h, x + s) for s in all_bits(m))
        assert (abs(best) < 1e-9) == (2 * x[0] + 3 * x[1] <= 3)


@pytest.mark.parametrize("encoding", ["binary", "one_hot"])
def test_greater_equal(encoding):
    m = slack_size(5, encoding)
    with obs(2 + m) as q:
        expr = 2 * B(q[0]) + 3 * B(q[1])
        h = greater_equal(expr, 3, q[2:], encoding)
    for x in all_bits(2):
        best = min(_evaluate(h, x + s) for s in all_bits(m))
        assert (abs(best) < 1e-9) == (2 * x[0] + 3 * x[1] >= 3)


def test_functions_on_process_qubits():
    p = Process()
    q = p.alloc(3)
    h = exactly_one(q)
    assert h.is_bound


# --- QUBO class ---


def knapsack():
    values, weights = [3, 4, 5], [2, 3, 4]
    qubo = QUBO()
    x = qubo.alloc(3)
    qubo.maximize(sum(v * B(q) for v, q in zip(values, x)))
    qubo.less_equal(sum(w * B(q) for w, q in zip(weights, x)), 5, name="capacity")
    return qubo


def test_knapsack_slack_and_solution():
    qubo = knapsack()
    # slack in [0, 5] -> 3 qubits in binary encoding
    assert qubo.num_qubits == 6
    state, _ = exact_solver(qubo, qubo.num_qubits)
    assert qubo.is_feasible(state)
    assert qubo.evaluate(state) == -7
    assert f"{state:06b}"[:3] == "110"


def test_knapsack_one_hot_slack():
    values, weights = [3, 4, 5], [2, 3, 4]
    qubo = QUBO(3)
    x = qubo.qubits
    qubo.maximize(sum(v * B(q) for v, q in zip(values, x)))
    qubo.less_equal(
        sum(w * B(q) for w, q in zip(weights, x)), 5, encoding="one_hot"
    )
    assert qubo.num_qubits == 3 + 6
    state, _ = exact_solver(qubo, qubo.num_qubits)
    assert qubo.is_feasible(state)
    assert qubo.evaluate(state) == -7


def test_infeasible_states_have_higher_energy():
    qubo = knapsack()
    energies = {s: qubo.energy(s) for s in range(1 << qubo.num_qubits)}
    worst_feasible = max(e for s, e in energies.items() if qubo.is_feasible(s))
    best_infeasible = min(e for s, e in energies.items() if not qubo.is_feasible(s))
    assert worst_feasible < best_infeasible


def test_energy_matches_simulation():
    qubo = knapsack()
    for state in (0, 5, 37, 63):
        assert energy(qubo, state, qubo.num_qubits) == pytest.approx(
            qubo.energy(state)
        )


def test_one_hot_var():
    qubo = QUBO()
    v = qubo.one_hot_var([10, 20, 30], name="choice")
    qubo.minimize(-v)
    state, _ = exact_solver(qubo, qubo.num_qubits)
    assert qubo.evaluate(state, v) == 30
    assert qubo.violations("110") == ["choice"]


def test_binary_var_with_equality():
    qubo = QUBO()
    a = qubo.int_var(5)
    b = qubo.int_var(5)
    qubo.equal(a + b, 7)
    qubo.minimize(a)
    state, _ = exact_solver(qubo, qubo.num_qubits)
    assert qubo.evaluate(state, a) == 2
    assert qubo.evaluate(state, b) == 5


def test_redundant_inequality_adds_nothing():
    qubo = QUBO(2)
    qubo.less_equal(B(qubo.qubits[0]) + B(qubo.qubits[1]), 2)
    assert qubo.num_qubits == 2
    assert qubo.constraints == []


def test_tight_inequality_becomes_equality():
    qubo = QUBO(2)
    qubo.greater_equal(B(qubo.qubits[0]) + B(qubo.qubits[1]), 2)
    assert qubo.num_qubits == 2
    assert len(qubo.constraints) == 1


def test_impossible_inequality():
    qubo = QUBO(2)
    with pytest.raises(ValueError):
        qubo.less_equal(B(qubo.qubits[0]) + B(qubo.qubits[1]), -1)


def test_explicit_penalty_weight():
    qubo = QUBO(2)
    qubo.exactly_one(qubo.qubits, penalty=10)
    assert qubo.energy("11") == pytest.approx(10)


def test_rejects_bound_expression():
    p = Process()
    q = p.alloc(2)
    qubo = QUBO(2)
    with pytest.raises(ValueError):
        qubo.minimize(B(q[0]))


def test_rejects_non_diagonal_expression():
    qubo = QUBO(1)
    with obs():
        from ket import X

        h = X(qubo.qubits[0]) + 0
    with pytest.raises(ValueError):
        qubo.minimize(h)


def test_rejects_unknown_qubit():
    qubo = QUBO(1)
    with obs(3) as q:
        h = B(q[2])
    with pytest.raises(ValueError):
        qubo.minimize(h)


def test_bind_and_exp_value():
    qubo = knapsack()
    p = Process()
    q = p.alloc(qubo.num_qubits)
    assert exp_value(qubo(q)).get() == pytest.approx(qubo.energy(0))


# --- Operator API ---


def knapsack_ops():
    values, weights = [3, 4, 5], [2, 3, 4]
    qubo = QUBO()
    x = qubo.binary_vars(3)
    qubo -= sum(v * xi for v, xi in zip(values, x))
    qubo += (sum(w * xi for w, xi in zip(weights, x)) <= 5, "capacity")
    return qubo, x


def test_operator_knapsack_matches_method_api():
    qubo, _ = knapsack_ops()
    ref = knapsack()
    assert qubo.num_qubits == ref.num_qubits
    for state in range(1 << qubo.num_qubits):
        assert qubo.energy(state) == pytest.approx(ref.energy(state))
    assert [name for name, _, _ in qubo.constraints] == ["capacity"]


def test_expression_arithmetic():
    qubo = QUBO()
    x, y = qubo.binary_vars(2)
    e = 3 * x + y / 2 - 1
    assert isinstance(e, Expression)
    for bits in all_bits(2):
        assert qubo.evaluate(bits, e) == pytest.approx(3 * bits[0] + bits[1] / 2 - 1)
    assert isinstance(-x, Expression) and isinstance(1 - x, Expression)


def test_product_is_pauli_algebra():
    qubo = QUBO()
    x, y = qubo.binary_vars(2)
    for bits in all_bits(2):
        assert qubo.evaluate(bits, x * x) == pytest.approx(bits[0])
        assert qubo.evaluate(bits, x * y) == pytest.approx(bits[0] * bits[1])
        assert qubo.evaluate(bits, (x + y) ** 2) == pytest.approx(sum(bits) ** 2)
        assert qubo.evaluate(bits, (x + y) ** 0) == pytest.approx(1)


def test_mixing_with_hamiltonians():
    qubo = QUBO(2)
    x = qubo.binary_var()
    h = B(qubo.qubits[0])  # plain Hamiltonian
    for e in (h + x, x + h, h * x, x * h, h - x):
        assert isinstance(e, Expression)
    assert isinstance(h <= x, Constraint)


def test_constraint_normalization():
    qubo = QUBO()
    x, y = qubo.binary_vars(2)
    c = x + 1 <= y
    assert c.sense == "<=" and c.rhs == pytest.approx(-1)
    c = 2 >= x  # reflected: x <= 2
    assert c.sense == "<=" and c.rhs == pytest.approx(2)


def test_strict_comparisons():
    qubo = QUBO()
    x, y = qubo.binary_vars(2)
    assert (x + y < 2).rhs == pytest.approx(1)
    assert (x + y > 0).rhs == pytest.approx(1)
    with pytest.raises(ValueError):
        x + 0.5 * y < 1


@pytest.mark.parametrize(
    "build, feasible",
    [
        (lambda x, y: x + y == 1, lambda a, b: a + b == 1),
        (lambda x, y: x + y <= 1, lambda a, b: a + b <= 1),
        (lambda x, y: x + y >= 1, lambda a, b: a + b >= 1),
        (lambda x, y: x + y < 1, lambda a, b: a + b < 1),
        (lambda x, y: x + y > 1, lambda a, b: a + b > 1),
        (lambda x, y: 2 * x - y >= 1, lambda a, b: 2 * a - b >= 1),
    ],
)
def test_constraint_senses(build, feasible):
    qubo = QUBO()
    x, y = qubo.binary_vars(2)
    qubo += build(x, y)
    slack = qubo.num_qubits - 2
    for a, b in all_bits(2):
        ok = any(qubo.is_feasible([a, b] + s) for s in all_bits(slack))
        assert ok == feasible(a, b)


def test_tuple_options_and_list():
    qubo = QUBO()
    x = qubo.binary_vars(3)
    qubo += [(x[0] + x[1] == 1, 7.0, "first"), x[1] + x[2] <= 1]
    qubo += x[0] == 1, "third"
    names = [name for name, _, _ in qubo.constraints]
    weights = [w for _, _, w in qubo.constraints]
    assert names == ["first", "constraint 1", "third"]
    assert weights == [7.0, None, None]


def test_add_with_one_hot_slack():
    qubo = QUBO()
    x = qubo.binary_vars(3)
    qubo.add(x[0] + x[1] + x[2] <= 2, encoding="one_hot", name="cap")
    assert qubo.num_qubits == 3 + 3


def test_int_var_one_hot():
    qubo = QUBO()
    n = qubo.int_var(3, encoding="one_hot")
    assert qubo.num_qubits == 4 and len(qubo.constraints) == 1
    qubo += -n
    state, _ = exact_solver(qubo, qubo.num_qubits)
    assert qubo.evaluate(state, n) == 3


def test_int_var_keeps_qubits():
    qubo = QUBO()
    n = qubo.int_var(5)
    assert n.qubits.qubits == [0, 1, 2]


def test_constraint_has_no_truth_value():
    qubo = QUBO()
    x, y = qubo.binary_vars(2)
    with pytest.raises(TypeError):
        if x == y:
            pass
    with pytest.raises(TypeError):
        x != y


def test_expression_rejects_bound_and_non_diagonal():
    p = Process()
    q = p.alloc(1)
    with pytest.raises(ValueError):
        Expression(B(q))
    with obs(1) as v:
        from ket import X

        h = X(v[0]) + 0
    with pytest.raises(ValueError):
        Expression(h)


def test_reprs():
    qubo = QUBO()
    x, y = qubo.binary_vars(2)
    assert repr(3 * x - x * y + 2) == "<Ket 'Expression' 3*x0 - x0*x1 + 2>"
    assert repr(x + y <= 1) == "<Ket 'Constraint' x0 + x1 <= 1>"


def test_expression_with_exp_value():
    qubo = QUBO()
    x, y = qubo.binary_vars(2)
    p = Process()
    q = p.alloc(2)
    from ket import X as XGate

    XGate(q[1])
    assert exp_value(x + 2 * y, q).get() == pytest.approx(2)


def test_operator_assignment_example():
    cost = [[1, 3], [2, 1], [4, 2]]
    qubo = QUBO()
    x = [qubo.binary_vars(2) for _ in range(3)]
    qubo += sum(cost[t][m] * x[t][m] for t in range(3) for m in range(2))
    qubo += [(sum(x[t]) == 1, f"task {t}") for t in range(3)]
    qubo += sum(x[t][0] for t in range(3)) <= 1, "machine 0"
    state, _ = exact_solver(qubo, qubo.num_qubits)
    assert qubo.is_feasible(state)
    assert qubo.evaluate(state) == 4
