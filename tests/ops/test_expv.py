# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import cmath
from ket import (
    B,
    CNOT,
    H,
    ObsQubits,
    Process,
    obs,
    commutator,
    evolve,
    exp_value,
    X,
    Y,
    Z,
)


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


def test_exp_value_is_none_before_batch_execution():
    """In batch mode, ``value`` is ``None`` until ``get`` runs the process."""
    p = Process(execution="batch")
    a, b = p.alloc(2)
    CNOT(H(a), b)
    with obs():
        result = exp_value(Z(a) * Z(b) + 1)

    assert result.value is None
    assert result.get() == pytest.approx(2.0, abs=1e-5)
    assert result.value == pytest.approx(2.0, abs=1e-5)


# Unbound Hamiltonians built with ``obs(n)``


def test_obs_without_size_yields_none():
    """``with obs():`` keeps its old behaviour."""
    with obs() as q:
        assert q is None


def test_obs_qubits_indexing():
    """Placeholder qubits support slicing, iteration, reversal and concatenation."""
    with obs(4) as q:
        assert isinstance(q, ObsQubits)
        assert len(q) == 4
        assert q[1:3].qubits == [1, 2]
        assert [x.qubits for x in q] == [[0], [1], [2], [3]]
        assert reversed(q).qubits == [3, 2, 1, 0]
        assert (q[0] + q[3]).qubits == [0, 3]
        assert q.at([2, 0]).qubits == [2, 0]
        with pytest.raises(ValueError, match="overlapping"):
            _ = q[0] + q[0]


def test_unbound_hamiltonian_algebra():
    """Algebra works on unbound operators without any process."""
    with obs(2) as q:
        h = X(q[0]) * X(q[1]) + Z(q[0]) - 2
        comm = commutator(X(q[0]), Y(q[0]))

    assert not h.is_bound
    assert h.ket_process is None
    assert h.num_qubits == 2
    assert "unbound" in repr(h)
    assert len(comm.terms) == 1
    assert comm.terms[0].map == {0: "Z"}
    assert comm.terms[0].coef == 2j


def test_bind_maps_positions_to_process_qubits():
    """Placeholder ``i`` becomes ``qubits[i]``, also for non-contiguous qubits."""
    with obs(2) as q:
        h = X(q[0]) * Z(q[1]) + 0.5 * Y(q[1])

    p = Process()
    r = p.alloc(4)
    target = r[3] + r[1]
    hb = h.bind(target)

    assert hb.is_bound
    assert hb.ket_process is p
    assert hb.terms[0].map == {3: "X", 1: "Z"}
    assert hb.terms[1].map == {1: "Y"}
    assert hb.terms[1].coef == 0.5
    # the call shorthand gives the same result
    assert [t.map for t in h(target).terms] == [t.map for t in hb.terms]
    # the unbound original is unchanged
    assert h.terms[0].map == {0: "X", 1: "Z"}


def test_bind_errors():
    """Binding checks size and binding state."""
    with obs(3) as q:
        h = Z(q[0]) * Z(q[2])

    p = Process()
    with pytest.raises(ValueError, match="needs 3 qubits"):
        h.bind(p.alloc(2))

    hb = h(p.alloc(3))
    with pytest.raises(ValueError, match="already bound"):
        hb.bind(p.alloc(3))

    with pytest.raises(TypeError):
        h.bind(q)


def test_mixing_bound_and_unbound_raises():
    """Bound and unbound operators cannot be combined."""
    p = Process()
    r = p.alloc(1)
    with obs(1) as q:
        with pytest.raises(ValueError, match="bound and unbound"):
            _ = X(q) + X(r)
        with pytest.raises(ValueError, match="bound and unbound"):
            _ = Z(q[0]) * X(r)


def test_exp_value_unbound():
    """``exp_value(h, q)`` and ``exp_value(h(q))`` agree with the bound version."""
    with obs(2) as q:
        h = X(q[0]) * X(q[1]) + Y(q[0]) * Y(q[1]) + Z(q[0]) * Z(q[1]) + 1

    p = Process()
    r = p.alloc(2)
    CNOT(H(r[0]), r[1])  # |Φ+⟩: <XX> = 1, <YY> = -1, <ZZ> = 1

    assert exp_value(h, r).get() == pytest.approx(2.0, abs=1e-5)
    assert exp_value(h(r)).get() == pytest.approx(2.0, abs=1e-5)


def test_exp_value_unbound_errors():
    """Unbound needs qubits; bound refuses them."""
    with obs(1) as q:
        h = Z(q)

    p = Process()
    r = p.alloc(1)
    with pytest.raises(ValueError, match="unbound Hamiltonian"):
        exp_value(h)
    with pytest.raises(ValueError, match="already bound"):
        exp_value(h(r), r)


def test_evolve_unbound():
    """``evolve(h, q)`` binds before evolving."""
    with obs(1) as q:
        h = cmath.pi / 2 * X(q)  # RX(pi): |0> -> |1>

    p = Process()
    r = p.alloc(2)
    evolve(h, r[1])
    with obs():
        assert exp_value(Z(r[1])).get() == pytest.approx(-1.0, abs=1e-5)
        assert exp_value(Z(r[0])).get() == pytest.approx(1.0, abs=1e-5)

    with pytest.raises(ValueError, match="unbound Hamiltonian"):
        evolve(h)


def test_unbound_qubo_with_b():
    """``B`` works on placeholder qubits."""
    with obs(2) as q:
        cost = B(q[0]) + 2 * B(q[1])

    p = Process()
    r = p.alloc(2)
    X(r[1])
    assert exp_value(cost, r).get() == pytest.approx(2.0, abs=1e-5)


def test_unbound_as_callable_for_qulib():
    """An unbound Hamiltonian can stand in for ``Callable[[Quant], Hamiltonian]``."""
    from ket.qulib import pauli_string

    with obs(3) as q:
        h = 0.5 * Z(q[0]) * X(q[2]) - Y(q[1])

    assert pauli_string(h, 3) == [("ZIX", 0.5), ("IYI", -1.0)]


def test_addition_does_not_modify_operands():
    """Adding Hamiltonians or Paulis returns new terms and leaves the inputs alone."""
    p = Process()
    q = p.alloc(2)
    with obs():
        px = X(q[0])
        h = px + Z(q[1])

    h2 = h + h
    h3 = px + px

    assert [t.coef for t in h.terms] == [1.0, 1.0]
    assert [t.coef for t in h2.terms] == [2.0, 2.0]
    assert px.coef == 1.0
    assert h3.terms[0].coef == 2.0

    _ = h - h  # cancels to zero; must not zero out h
    assert [t.coef for t in h.terms] == [1.0, 1.0]


def test_pauli_string_accepts_single_pauli():
    """``pauli_string`` works when the callable returns one ``Pauli`` term."""
    from ket.qulib import pauli_string

    with obs(3) as q:
        h = 0.5 * Z(q[0]) * X(q[2])

    assert pauli_string(h, 3) == [("ZIX", 0.5)]

    def single_term(r):
        with obs():
            return 0.5 * Z(r[0]) * X(r[2])

    assert pauli_string(single_term, 3) == [("ZIX", 0.5)]
