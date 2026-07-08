# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import cmath
import math
from ket import Process, dump
from ket.qulib.prepare import bell, ghz, w, dicke, pauli_state


def test_prepare_bell():
    """Test the creation of the Bell state |00> + |11>."""
    p = Process()
    q = p.alloc(2)
    bell(q[0], q[1])

    state = dump(q)

    probs = state.probability
    assert 0 in probs and 3 in probs
    assert len(probs) == 2
    assert cmath.isclose(probs[0], 0.5, abs_tol=1e-5)


def test_prepare_ghz():
    """Test the creation of the GHZ state |000> + |111>."""
    p = Process()
    q = p.alloc(3)
    ghz(q)

    state = dump(q)

    probs = state.probability
    assert 0 in probs and 7 in probs
    assert len(probs) == 2
    assert cmath.isclose(probs[7], 0.5, abs_tol=1e-5)


def test_prepare_w():
    """Test the creation of the W state |001> + |010> + |100>."""
    p = Process()
    q = p.alloc(3)
    w(q)

    state = dump(q)

    probs = state.probability
    # W state has 1 excitation: states 1 (001), 2 (010), 4 (100)
    expected_states = {1, 2, 4}
    for s in expected_states:
        assert s in probs
        assert cmath.isclose(probs[s], 1 / 3, abs_tol=1e-5)


def test_prepare_dicke():
    """Test the creation of a Dicke state with specific Hamming weight."""
    p = Process()

    n_qubits = 4
    excitations = 2

    q = p.alloc(n_qubits)
    dicke(excitations, q)

    state = dump(q)

    probs = state.probability

    # 4 choose 2 = 6 possible states
    assert len(probs) == 6

    # Every state must have exactly 2 ones in its binary representation
    for s, prob in probs.items():
        assert s.bit_count() == excitations
        assert cmath.isclose(prob, 1 / 6, abs_tol=1e-5)


def test_prepare_pauli_states():
    """Test preparation of Pauli +1 and -1 eigenstates."""
    p = Process()
    q = p.alloc(1)

    # Prepare |-i> (Y eigenstate with -1 eigenvalue)
    pauli_state("Y", -1, q)

    state = dump(q)

    amplitudes = state.get()
    # |-i> = 1/sqrt(2) * (|0> - i|1>)
    assert cmath.isclose(amplitudes[0].real, 1 / math.sqrt(2), abs_tol=1e-5)
    assert cmath.isclose(amplitudes[1].imag, -1 / math.sqrt(2), abs_tol=1e-5)
