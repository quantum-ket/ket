# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Soundness tests for the compute-action-uncompute verification.

Covers the verification performed by ``around`` and ``undo`` on auxiliary
qubits, and the approximated (relative-phase) decomposition enabled by
``around``.
"""

import pytest

from ket import (
    CNOT,
    H,
    P,
    RY,
    Process,
    QuantumState,
    X,
    Z,
    around,
    ctrl,
    undo,
    using_aux,
)

np = pytest.importorskip("numpy")
from ket.npsim import NPSim  # pylint: disable=wrong-import-position


def _fidelity(a, b):
    return abs(sum(a.get(k, 0).conjugate() * b.get(k, 0) for k in set(a) | set(b)))


def test_around_rejects_unitary_on_aux():
    p = Process()
    aux = p.alloc_aux()
    with pytest.raises(RuntimeError):
        with around(H, aux):
            pass


def test_around_rejects_permutation_on_aux_in_action():
    p = Process()
    q = p.alloc()
    aux = p.alloc_aux()
    with pytest.raises(RuntimeError):
        with around(ctrl(q, X), aux):
            X(aux)


def test_around_rejects_permutation_on_read_qubit_in_action():
    p = Process()
    q = p.alloc()
    aux = p.alloc_aux()
    with pytest.raises(RuntimeError):
        with around(ctrl(q, X), aux):
            X(q)


def test_around_rejects_unitary_on_read_qubit_in_compute():
    """A qubit read by an auxiliary qubit must not leave the computational basis."""
    p = Process()
    q = p.alloc()
    aux = p.alloc_aux()

    def compute():
        CNOT(q, aux)
        H(q)

    with pytest.raises(RuntimeError):
        with around(compute, ket_process=p):
            Z(q)


def test_undo_rejects_unitary_on_read_qubit_in_compute():
    p = Process()
    q = p.alloc()
    aux = p.alloc_aux()

    def compute(a):
        CNOT(q, a)
        H(q)

    with pytest.raises(RuntimeError):
        undo(compute, aux)


def test_around_blocked_set_is_transitive():
    """If aux reads q and q reads r, r must also be blocked."""
    p = Process()
    q, r = p.alloc(2)
    aux = p.alloc_aux()

    def compute():
        CNOT(q, aux)
        CNOT(r, q)

    with pytest.raises(RuntimeError):
        with around(compute, ket_process=p):
            H(r)


def test_around_accepts_diagonal_on_blocked_qubit():
    p = Process(NPSim(2))
    q = p.alloc()
    aux = p.alloc_aux()
    H(q)
    with around(ctrl(q, X), aux):
        Z(q)
    state = QuantumState(q + aux).get()
    assert all(k & 1 == 0 for k, a in state.items() if abs(a) > 1e-9)


@pytest.mark.parametrize("num_controls", [2, 3])
@pytest.mark.parametrize("action", [X, H, Z])
def test_approximated_decomposition_with_action_on_control(num_controls, action):
    """The approximation must not be used when the action changes a control."""

    def build(use_around):
        p = Process(NPSim(num_controls + 1))
        c = p.alloc(num_controls)
        t = p.alloc()
        H(c)
        P(0.3, c[0])
        RY(0.7, t)
        if use_around:
            with around(ctrl(c, X), t):
                action(c[0])
        else:
            ctrl(c, X)(t)
            action(c[0])
            ctrl(c, X)(t)
        return QuantumState(c + t).get()

    assert _fidelity(build(False), build(True)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "num_controls,free_qubits", [(2, 0), (3, 0), (4, 0), (5, 2), (7, 2)]
)
def test_approximated_decomposition_uncompute_is_adjoint(num_controls, free_qubits):
    """With a diagonal action (Theorem 2), the result must be exact."""

    def build(use_around):
        p = Process(NPSim(num_controls + 1 + free_qubits))
        c = p.alloc(num_controls)
        t = p.alloc()
        for i, x in enumerate(c):
            RY(0.3 + 0.2 * i, x)
        RY(0.7, t)
        if use_around:
            with around(ctrl(c, X), t):
                Z(t)
        else:
            ctrl(c, X)(t)
            Z(t)
            ctrl(c, X)(t)
        return QuantumState(c + t).get()

    assert _fidelity(build(False), build(True)) == pytest.approx(1.0)


@using_aux(aux=1)
def _tof5(a, b, c, d, e, aux):
    with around(ctrl(a + b + c, X), aux):
        ctrl(aux + d, X)(e)


@pytest.mark.parametrize("basis_state", range(32))
def test_toffoli5_is_exact(basis_state):
    """The 18-CNOT Toffoli-5 must equal the exact 4-controlled X."""

    def column(gate):
        p = Process(NPSim(6))
        q = p.alloc(5)
        for i in range(5):
            if (basis_state >> (4 - i)) & 1:
                X(q[i])
        gate(q)
        return QuantumState(q).get()

    exact = column(lambda q: ctrl(q[:4], X)(q[4]))
    approx = column(lambda q: _tof5(*[q[i] for i in range(5)]))
    for k in set(exact) | set(approx):
        assert approx.get(k, 0) == pytest.approx(exact.get(k, 0))


def test_block_builder_recovers_from_libket_error():
    """A Libket error while finalizing a controlled block (control-target
    conflict) inside an ``around`` must raise cleanly and leave the process
    usable. Before the fix, the failed block stayed on the builder stack and
    the next append crashed the interpreter (segmentation fault)."""
    from ket import control  # pylint: disable=import-outside-toplevel
    from ket.clib.wrapper import LibketError  # pylint: disable=import-outside-toplevel

    p = Process(NPSim(4))
    q = p.alloc(3)
    aux = p.alloc_aux()
    with pytest.raises(LibketError):
        with around(ctrl(q[0], X), aux):
            with control(q[1] + q[2]):
                H(q[1])  # control-target conflict
    X(q[2])
    assert not p._blocks  # pylint: disable=protected-access
    state = QuantumState(q).get()
    assert len(state) == 1
