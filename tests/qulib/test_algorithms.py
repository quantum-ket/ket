# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import math
import random
import ket
from ket import (
    Process,
    H,
    X,
    Y,
    Z,
    CZ,
    around,
    cat,
    dump,
    obs,
    P,
    adj,
    measure,
    control,
)
from ket.qulib import oracle, exact_solver, simulated_annealing, energy
from ket.gates import QFT

# --- Integration Helpers for Algorithms ---


def run_grover(size: int, oracle_func: callable, outcomes: int = 1) -> int:
    """Helper implementing Grover's algorithm to test the entire stack."""
    p = Process(simulator="dense", num_qubits=size)
    s = H(p.alloc(size))

    steps = int((math.pi / 4) * math.sqrt(2**size / outcomes))
    for _ in range(steps):
        oracle_func(s)
        with around(cat(H, X), s):
            CZ(*s)

    # Using dump to ensure probability is peaked at the correct state instead of flaky measures
    state = dump(s)

    # Return the most probable state
    return max(state.probability.items(), key=lambda item: item[1])[0]


def run_phase_estimator(oracle_gate: callable, precision: int) -> float:
    """Helper implementing Phase Estimation to test QFT and controlled scopes."""
    p = Process(simulator="dense", num_qubits=precision + 1)
    ctr = H(p.alloc(precision))
    tgr = X(p.alloc())

    for i, c in enumerate(ctr):
        with control(c):
            oracle_gate(i, tgr)

    adj(QFT)(ctr, do_swap=True)

    # Measuring for estimation
    res = measure(reversed(ctr))

    return res.value / (2**precision)


def custom_phase_oracle(phase_val: float, i: int, tgr):
    """Oracle that applies a phase shift for Phase Estimation."""
    P(2 * math.pi * phase_val * (2**i), tgr)


# --- Tests ---


def test_grover_search():
    """Integration test: Grover's search algorithm using the native phase_oracle."""
    size = 6
    target = random.randint(0, (1 << size) - 1)

    found = run_grover(size, oracle.phase_oracle(target))
    assert found == target


def test_phase_estimation():
    """Integration test: Phase Estimation using native QFT and adj modifier."""
    precision = 10
    expected_phase = 0.3125  # Exactly representable in binary fractions

    oracle_func = lambda i, t: custom_phase_oracle(expected_phase, i, t)

    estimated = run_phase_estimator(oracle_func, precision)
    assert math.isclose(estimated, expected_phase, abs_tol=1e-3)


def test_energy_evaluation():
    """Test the native energy evaluation helper function."""

    def simple_hamiltonian(qubits):
        with obs():
            return Z(qubits[0]) * 2.0

    # For state |0>, Z has eigenvalue 1 -> Energy = 2.0
    energy_0 = energy(simple_hamiltonian, state="0", num_qubits=1)
    assert math.isclose(energy_0, 2.0, abs_tol=1e-5)

    # For state |1>, Z has eigenvalue -1 -> Energy = -2.0
    energy_1 = energy(simple_hamiltonian, state="1", num_qubits=1)
    assert math.isclose(energy_1, -2.0, abs_tol=1e-5)


def test_exact_solver_and_simulated_annealing():
    """Test solving a small optimization problem with both exact and heuristic solvers."""

    # Construct a simple QUBO/Ising Hamiltonian where the ground state is |101>
    def problem_hamiltonian(q):
        # Minimized when q[0]=1 (-Z), q[1]=0 (+Z), q[2]=1 (-Z)
        with obs():
            return Z(q[0]) - Z(q[1]) + Z(q[2])

    num_qubits = 3
    # Decimal 5 is 101 in binary
    expected_state = 5
    expected_energy = -3.0

    # 1. Exact Solver
    exact_state, exact_en = exact_solver(problem_hamiltonian, num_qubits)
    assert exact_state == expected_state
    assert math.isclose(exact_en, expected_energy, abs_tol=1e-5)

    # 2. Simulated Annealing
    # Multiprocessing disabled here to keep test overhead minimal and deterministic
    sa_state, sa_en = simulated_annealing(
        problem_hamiltonian,
        num_qubits,
        initial_temp=10.0,
        num_evaluations=5,
        multiprocessing=False,
    )
    # Since SA is heuristic, it might not find the global minimum every single time in 5 shots,
    # but for this trivial 3-qubit convex landscape it should essentially always hit it.
    assert sa_state == expected_state
    assert math.isclose(sa_en, expected_energy, abs_tol=1e-5)
