# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from ket import Process, measure, sample, X, using_aux


def test_measurement_basic():
    """Test simple deterministic measurement."""
    p = Process()
    q = p.alloc(3)

    X(q[1])  # Prepare state |010> (which is 2 in decimal)

    res = measure(q)

    assert res.value == 2
    assert res.raw_value == 2
    assert res.bitstring == "010"


def test_sample_basic():
    """Test basic sample execution."""
    p = Process()
    q = p.alloc(2)

    X(q[0])  # Prepare state |10> (which is 2 in decimal)

    res = sample(q, shots=100)

    samples = res.value
    assert samples is not None
    assert samples[2] == 100
    assert res.most_frequent_state() == 2


def test_measurement_aux_qubits_blocked():
    """Verify that safe auxiliary qubits cannot be measured or sampled."""
    p = Process()
    q = p.alloc(1)
    aux = p.alloc_aux(1)

    # Trying to measure an auxiliary qubit should raise ValueError
    with pytest.raises(ValueError, match="Auxiliary qubits cannot be measured"):
        measure(aux)

    # Trying to sample an auxiliary qubit should also raise ValueError
    with pytest.raises(ValueError, match="Auxiliary qubits cannot be measured"):
        sample(aux)


def test_measurement_unsafe_aux():
    """Verify that auxiliary qubits CAN be measured if unsafe mode is enabled."""

    @using_aux(unsafe=True, aux=1)
    def unsafe_measurement(q, aux):
        # This should NOT raise an exception because unsafe=True bypasses the check
        X(aux)
        return measure(aux)

    p = Process()
    q = p.alloc(1)

    res = unsafe_measurement(q)

    assert res.value == 1
