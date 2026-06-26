# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import cmath
from ket import Process, obs, commutator, exp_value, X, Y, Z


def test_pauli_algebra_multiplication():
    """Test the algebraic multiplication rules of Pauli matrices."""
    p = Process()
    q = p.alloc(1)

    with obs():
        px = X(q)
        py = Y(q)

    # X * Y = iZ
    pxy = px @ py
    assert pxy.map[q.qubits[0]] == "Z"
    assert pxy.coef == 1j

    # Y * X = -iZ
    pyx = py @ px
    assert pyx.map[q.qubits[0]] == "Z"
    assert pyx.coef == -1j

    # X * X = I
    pxx = px @ px
    assert pxx.map[q.qubits[0]] == "I"
    assert pxx.coef == 1.0


def test_hamiltonian_addition_and_filtering():
    """Test if Hamiltonians correctly add up and filter out zeroed terms."""
    p = Process()
    q = p.alloc(1)

    with obs():
        px = X(q)
        pz = Z(q)

    # Create Hamiltonians
    h1 = px + pz
    h2 = px - pz

    # (X + Z) + (X - Z) = 2X
    h_sum = h1 + h2
    assert len(h_sum.terms) == 1
    assert h_sum.terms[0].map[q.qubits[0]] == "X"
    assert h_sum.terms[0].coef == 2.0


def test_commutator():
    """Test the commutator [X, Y] = 2iZ."""
    p = Process()
    q = p.alloc(1)

    with obs():
        px = X(q)
        py = Y(q)

    comm = commutator(px, py)  # XY - YX = iZ - (-iZ) = 2iZ

    assert len(comm.terms) == 1
    assert comm.terms[0].map[q.qubits[0]] == "Z"
    assert comm.terms[0].coef == 2j


def test_exp_value_execution():
    """Test the execution and extraction of expected values."""
    p = Process()
    q = p.alloc(1)

    # State is |1>
    X(q)

    # The expected value of Z for |1> is -1
    with obs():
        pz = Z(q)

    ev = exp_value(pz)

    assert cmath.isclose(ev.value, -1.0, abs_tol=1e-5)


def test_exp_value_unsupported_complex():
    """Test that calculating expected values for complex coefficients fails."""
    p = Process()
    q = p.alloc(1)

    with obs():
        px = X(q)

    # Give it a complex coefficient
    complex_h = px * 1j

    with pytest.raises(ValueError, match="Complex coefficients are not supported"):
        _ = exp_value(complex_h)
