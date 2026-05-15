# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import math
import ket
from ket.npsim import NPSim

def test_npsim_bell_state_measurement():
    """Test Bell state preparation and measurement on NPSim."""
    sim = NPSim(num_qubits=2)
    p = ket.Process(execution_target=sim)

    q = p.alloc(2)
    ket.CNOT(ket.H(q[0]), q[1])

    res = ket.measure(q)
    p.execute()

    # Bell state must collapse to either |00> (0) or |11> (3)
    assert res.value in [0, 3]


def test_npsim_dump():
    """Test state vector dumping using NPSim."""
    sim = NPSim(num_qubits=1)
    p = ket.Process(execution_target=sim)

    q = p.alloc(1)
    ket.X(q)  # State |1>

    state = ket.dump(q)
    p.execute()

    probs = state.probability
    assert 1 in probs
    assert math.isclose(probs[1], 1.0, abs_tol=1e-5)


def test_npsim_rotations():
    """Test phase and rotation gates on NPSim."""
    sim = NPSim(num_qubits=1)
    p = ket.Process(execution_target=sim)

    q = p.alloc(1)
    ket.RX(math.pi, q)  # Equivalent to -iX

    state = ket.dump(q)
    p.execute()

    amplitudes = state.get()
    assert 1 in amplitudes
    # Amplitude should be roughly -i
    assert math.isclose(amplitudes[1].imag, -1.0, abs_tol=1e-5)
