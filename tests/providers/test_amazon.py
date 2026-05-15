# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2026 Ruan Luiz Molgero Lopes <ruan.molgero@grad.ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import math
import random
from functools import partial
import ket

# Gracefully skip these tests if the Amazon Braket SDK is not installed
braket = pytest.importorskip("braket.aws", reason="Amazon Braket SDK is not installed")
from ket.amazon import AmazonBraket


def grover(size: int, oracle_func: callable, outcomes: int = 1) -> int:
    """Helper to run Grover's search algorithm on the Amazon Braket backend."""
    device = AmazonBraket()
    p = ket.Process(execution_target=device)

    s = ket.H(p.alloc(size))

    steps = int((math.pi / 4) * math.sqrt(2**size / outcomes))
    for _ in range(steps):
        oracle_func(s)
        with ket.around(ket.cat(ket.H, ket.X), s):
            ket.CZ(*s)

    return ket.sample(s).most_frequent_state()


def custom_phase_oracle(phase_val: float, i: int, tgr: ket.Quant):
    """Performs a phase shift operation on the target qubit based on a given phase."""
    ket.P(2 * math.pi * phase_val * (2**i), tgr)


def phase_estimator(oracle_gate: callable, precision: int) -> float:
    """Helper to estimate the phase of an oracle on the Amazon Braket backend."""
    device = AmazonBraket()
    p = ket.Process(execution_target=device)

    ctr = ket.H(p.alloc(precision))
    trg = ket.X(p.alloc())

    for i, c in enumerate(ctr):
        with ket.control(c):
            oracle_gate(i, trg)

    ket.adj(ket.QFT)(ctr)

    return ket.sample(reversed(ctr)).most_frequent_state() / (2**precision)


def quantum_sum(a: int, b: int, size: int) -> int:
    """Helper to perform quantum addition using Qint on the Amazon Braket backend."""
    device = AmazonBraket()
    p = ket.Process(execution_target=device)

    qa = p.alloc(size + 1)
    qb = p.alloc(size - 1)

    # Use Qint to perform in-place quantum addition
    qi_a = ket.qint.Qint(qa, a)
    qi_b = ket.qint.Qint(qb, b)
    qi_a += qi_b

    return ket.sample(qa).most_frequent_state()


# --- Tests ---


def test_amazon_grover_search():
    """Test Grover's search algorithm execution on the Braket local simulator."""
    SIZE = 5
    NUM_EXECUTIONS = 3
    SUCCESS_THRESHOLD = 0.50

    looking_for = random.randint(0, pow(2, SIZE) - 1)
    results = []

    for _ in range(NUM_EXECUTIONS):
        results.append(grover(SIZE, ket.qulib.oracle.phase_oracle(looking_for)))

    # Statistical evaluation since local simulator uses finite shots
    success_count = results.count(looking_for)
    rate = success_count / NUM_EXECUTIONS
    assert rate >= SUCCESS_THRESHOLD


def test_amazon_phase_estimator():
    """Test the Phase Estimation algorithm execution on the Braket local simulator."""
    phase = 0.5791015625  # Exactly representable in binary
    estimate_pi = partial(phase_estimator, partial(custom_phase_oracle, phase))

    result = estimate_pi(10)
    assert math.isclose(result, phase, abs_tol=1e-5)


def test_amazon_quantum_adder():
    """Test quantum arithmetic (Qint addition) execution on the Braket local simulator."""
    SIZE = 5

    for _ in range(5):
        a = random.randint(0, pow(2, SIZE) - 1)
        b = random.randint(0, pow(2, SIZE - 1) - 1)

        expected_result = a + b
        result = quantum_sum(a, b, SIZE)

        assert result == expected_result
