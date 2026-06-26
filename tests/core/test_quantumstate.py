# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import cmath
from ket import Process, dump, H, X, CNOT


def test_quantumstate_dump_basic():
    """Test basic state dumping for a Bell state."""
    p = Process()
    a, b = p.alloc(2)

    # Prepare Bell state |00> + |11>
    CNOT(H(a), b)

    state = dump(a + b)

    states = state.get()

    # We expect states 0 (0b00) and 3 (0b11) to be present
    assert 0 in states
    assert 3 in states
    assert 1 not in states
    assert 2 not in states

    # Amplitudes should be roughly 1/sqrt(2)
    assert cmath.isclose(abs(states[0]), 1 / cmath.sqrt(2).real, abs_tol=1e-5)
    assert cmath.isclose(abs(states[3]), 1 / cmath.sqrt(2).real, abs_tol=1e-5)


def test_quantumstate_probabilities():
    """Test if probabilities are calculated correctly without lambda overhead."""
    p = Process()
    q = p.alloc(1)

    H(q)  # Superposition |0> and |1>
    state = dump(q)

    probs = state.probability
    assert probs is not None
    assert cmath.isclose(probs[0], 0.5, abs_tol=1e-5)
    assert cmath.isclose(probs[1], 0.5, abs_tol=1e-5)


def test_quantumstate_sample():
    """Test the fixed sample method using Counter to ensure correct distribution."""
    p = Process()
    q = p.alloc(1)

    H(q)
    state = dump(q)

    # Sample 10000 shots
    shots = 10000
    samples = state.sample(shots=shots, seed=42)

    assert samples is not None
    assert 0 in samples
    assert 1 in samples
    assert sum(samples.values()) == shots

    # Both 0 and 1 should have roughly 5000 counts each (~ 5% tolerance)
    assert 4500 < samples[0] < 5500
    assert 4500 < samples[1] < 5500


def test_quantumstate_show_modes():
    """Test the string and latex formatting of the quantum state."""
    p = Process()
    q = p.alloc(1)
    X(q)  # State |1>

    state = dump(q)

    str_out = state.show(mode="str")
    assert "|1⟩" in str_out
    assert "(100.00%)" in str_out
