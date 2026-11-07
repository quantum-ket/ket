# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

import random

import pytest

from ket import Process, X, Y, Z, H, B, obs, exp_value
from ket.qulib import energy, exact_solver, parallel_tempering, simulated_annealing
from ket.qulib.qubo import QUBO


def random_ising(n, seed):
    """Unbound all-to-all Ising Hamiltonian, built once."""
    rng = random.Random(seed)
    with obs(n) as q:
        h = sum(rng.uniform(-1, 1) * Z(q[i]) for i in range(n))
        h += sum(
            rng.uniform(-1, 1) * Z(q[i]) * Z(q[j])
            for i in range(n)
            for j in range(i + 1, n)
        )
        return h + 1.5


def simulated_energy(hamiltonian, state, n):
    """Reference: <s|H|s> on the sparse simulator."""
    p = Process(execution="live", simulator="sparse", num_qubits=n)
    q = p.alloc(n)
    bits = f"{state:0{n}b}"
    for i, b in enumerate(bits):
        if b == "1":
            X(q[i])
    return exp_value(hamiltonian(q)).get()


def test_energy_matches_simulator_including_off_diagonal_terms():
    n = 4

    def hamiltonian(q):
        with obs():
            return (
                2 * Z(q[0]) * Z(q[3])
                - 0.5 * Z(q[1])
                + 3 * X(q[2])  # zero diagonal
                + 0.25 * Y(q[0]) * Z(q[1])  # zero diagonal
                + 1
            )

    for state in range(1 << n):
        assert energy(hamiltonian, state, n) == pytest.approx(
            simulated_energy(hamiltonian, state, n)
        )


def test_energy_state_formats_agree():
    h = random_ising(3, 1)
    assert energy(h, 5, 3) == pytest.approx(energy(h, "101"))
    assert energy(h, [1, 0, 1]) == pytest.approx(energy(h, "101"))


def test_energy_accepts_unbound_hamiltonian():
    with obs(3) as v:
        h = Z(v[0]) - 2 * Z(v[1]) * Z(v[2])
    assert energy(h, "100") == pytest.approx(-1 - 2)


def test_energy_rejects_qubit_beyond_num_qubits():
    with obs(3) as v:
        h = Z(v[2])
    with pytest.raises(ValueError):
        energy(h, "10")


def test_energy_rejects_complex_coefficients():
    with obs(1) as v:
        h = 1j * Z(v[0])
    with pytest.raises(ValueError):
        energy(h, "1")


@pytest.mark.parametrize("n, seed", [(1, 0), (5, 1), (10, 2), (12, 3)])
def test_parallel_tempering_finds_exact_ground_state(n, seed):
    h = random_ising(n, seed)
    _, exact = exact_solver(h, n)
    state, value = parallel_tempering(h, n, seed=seed)
    assert value == pytest.approx(exact)
    assert energy(h, state, n) == pytest.approx(value)


def test_parallel_tempering_on_qubo():
    values, weights = [3, 4, 5, 6, 2], [2, 3, 4, 5, 1]
    qubo = QUBO()
    x = qubo.binary_vars(5)
    qubo -= sum(v * xi for v, xi in zip(values, x))
    qubo += sum(w * xi for w, xi in zip(weights, x)) <= 7
    state, value = parallel_tempering(qubo, qubo.num_qubits, seed=1)
    assert qubo.is_feasible(state)
    assert value == pytest.approx(exact_solver(qubo, qubo.num_qubits)[1])
    assert qubo.evaluate(state) == -10  # items worth 5 + 3 + 2


def test_parallel_tempering_is_reproducible():
    n = 60
    h = random_ising(n, 9)
    runs = [
        parallel_tempering(h, n, num_sweeps=50, seed=123, num_threads=t)
        for t in (1, 4, None)
    ]
    assert runs[0] == runs[1] == runs[2]


def test_parallel_tempering_many_qubits():
    # Above 64 qubits: states do not fit in a machine word.
    n = 100

    def hamiltonian(q):
        with obs():
            return sum(Z(q[i]) * Z(q[i + 1]) for i in range(n - 1))

    state, value = parallel_tempering(hamiltonian, n, seed=5)
    assert value == pytest.approx(-(n - 1))
    bits = f"{state:0{n}b}"
    assert all(a != b for a, b in zip(bits, bits[1:]))


def test_parallel_tempering_constant_hamiltonian():
    with obs(2) as v:
        h = Z(v[0]) - Z(v[0]) + 4
    assert parallel_tempering(h, 2, seed=0) == (0, pytest.approx(4))


def test_parallel_tempering_explicit_temperatures():
    h = random_ising(6, 4)
    exact = exact_solver(h, 6)[1]
    _, value = parallel_tempering(h, 6, min_temp=0.05, max_temp=5.0, seed=2)
    assert value == pytest.approx(exact)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_replicas=0),
        dict(num_sweeps=-1),
        dict(min_temp=0.0),
        dict(max_temp=-1.0),
        dict(min_temp=2.0, max_temp=1.0),
        dict(num_threads=0),
    ],
)
def test_parallel_tempering_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        parallel_tempering(random_ising(2, 0), 2, **kwargs)


def test_simulated_annealing_uses_rust_backend():
    h = random_ising(8, 6)
    exact = exact_solver(h, 8)[1]
    _, value = simulated_annealing(h, 8, num_evaluations=4)
    assert value == pytest.approx(exact)
    # Old keyword still accepted.
    _, value = simulated_annealing(h, 8, multiprocessing=False, final_temp=0.0)
    assert value == pytest.approx(exact)


# --- Exact solver (Rust) ---


def test_exact_solver_matches_brute_force_energy():
    n = 10
    h = random_ising(n, 11)
    state, value = exact_solver(h, n)
    assert value == pytest.approx(min(energy(h, s, n) for s in range(1 << n)))
    assert energy(h, state, n) == pytest.approx(value)


def test_exact_solver_20_qubits_and_threads():
    n = 20
    h = random_ising(n, 12)
    one = exact_solver(h, n, num_threads=1)
    many = exact_solver(h, n)
    assert one == many
    assert energy(h, one[0], n) == pytest.approx(one[1])


def test_exact_solver_limits():
    with pytest.raises(ValueError):
        exact_solver(random_ising(2, 0), 49)
    with pytest.raises(ValueError):
        exact_solver(random_ising(2, 0), 2, num_threads=0)


# --- Search options ---


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(adaptive_temps=False),
        dict(patience=30),
        dict(num_restarts=3, num_sweeps=200),
    ],
)
def test_parallel_tempering_options(kwargs):
    h = random_ising(12, 13)
    exact = exact_solver(h, 12)[1]
    state, value = parallel_tempering(h, 12, seed=4, **kwargs)
    assert value == pytest.approx(exact)
    assert energy(h, state, 12) == pytest.approx(value)


def test_patience_and_restarts_are_reproducible():
    h = random_ising(40, 14)
    kwargs = dict(seed=8, patience=25, num_restarts=2)
    assert parallel_tempering(h, 40, **kwargs) == parallel_tempering(h, 40, **kwargs)


@pytest.mark.parametrize("kwargs", [dict(patience=0), dict(num_restarts=0)])
def test_parallel_tempering_rejects_bad_options(kwargs):
    with pytest.raises(ValueError):
        parallel_tempering(random_ising(2, 0), 2, **kwargs)
