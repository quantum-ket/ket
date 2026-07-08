# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import cmath
from ket import Process, dump, H
from ket.qulib.oracle import xor_oracle, phase_oracle


def test_xor_oracle_behavior():
    """Test if the XOR oracle correctly maps |x>|0> to |x>|f(x)>."""
    p = Process()

    # Allocate 2 qubits for input (x) and 2 for output (y)
    x = p.alloc(2)
    y = p.alloc(2)

    # Create superposition on input
    H(x)

    # Define a simple function: f(x) = (x + 1) % 4
    def my_func(val):
        return (val + 1) % 4

    oracle = xor_oracle(my_func)
    oracle(x, y)

    state = dump(x + y)

    probs = state.probability

    # The states in the system should be |x>|f(x)>.
    # x=0 -> y=1 -> |0001> = 1
    # x=1 -> y=2 -> |0110> = 6
    # x=2 -> y=3 -> |1011> = 11
    # x=3 -> y=0 -> |1100> = 12
    expected_states = {1, 6, 11, 12}

    for s in expected_states:
        assert s in probs
        assert cmath.isclose(probs[s], 0.25, abs_tol=1e-5)


def test_phase_oracle_behavior():
    """Test if the phase oracle correctly flips the sign of the target state's amplitude."""
    p = Process()
    q = p.alloc(3)

    # Superposition of all 8 states
    H(q)

    target_state = 5  # 101 in binary

    # Apply phase oracle for the target state
    phase_oracle(target_state, q)

    state = dump(q)

    amplitudes = state.get()

    # Check that only the target state has a negative amplitude
    for s, amp in amplitudes.items():
        if s == target_state:
            assert cmath.isclose(amp.real, -1 / cmath.sqrt(8).real, abs_tol=1e-5)
        else:
            assert cmath.isclose(amp.real, 1 / cmath.sqrt(8).real, abs_tol=1e-5)
