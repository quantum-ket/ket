# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the gate-count optimizations of controlled blocks.

- ``control`` with two or more control qubits shares the controls through a
  clean auxiliary qubit (compute the AND, control the body by it, uncompute).
- Libket does not execute controlled gates that act as the identity on qubits
  known to be in |0> (``gate_count`` declares its inputs as unknown).
- Live execution keeps the executed gates fixed: gates appended after a
  measurement or dump are not simplified with them.

Each circuit is simulated with ``NPSim`` (Libket decomposition, live mode)
and compared with KBW (no decomposition).
"""

import pytest

from ket import H, P, Process, Qint, QuantumState, RX, RY, RZ, X, Z, around, control, ctrl
from ket.qulib import gate_count

np = pytest.importorskip("numpy")
from ket.npsim import NPSim  # pylint: disable=wrong-import-position


def _states(circuit, num_qubits, extra=2, kbw_extra=0):
    """Final states with NPSim and KBW; ``circuit`` may return extra qubits
    to include in the comparison."""
    states = []
    for process in (
        Process(NPSim(num_qubits + extra)),
        Process(simulator="dense", num_qubits=num_qubits + kbw_extra),
    ):
        q = process.alloc(num_qubits)
        for i, qubit in enumerate(q):
            RY(0.3 + 0.5 * i, qubit)
        more = circuit(q)
        dumped = q if more is None else q + more
        state = QuantumState(dumped).get()
        vector = np.zeros(2 ** len(dumped), dtype=complex)
        for index, amplitude in state.items():
            vector[index] = amplitude
        states.append(vector)
    return states


def _equal_up_to_phase(a, b):
    k = np.argmax(np.abs(b))
    phase = a[k] / b[k]
    return np.allclose(a, phase * b, atol=1e-6)


@pytest.mark.parametrize("num_controls", [2, 3])
def test_shared_control_is_exact(num_controls):
    def circuit(q):
        with control(q[:num_controls]):
            RX(0.4, q[num_controls])
            P(0.9, q[num_controls + 1])
            H(q[num_controls + 1])

    assert _equal_up_to_phase(*_states(circuit, num_controls + 2))


def test_shared_control_reduces_cnots():
    def add(c, e):
        with control(c):
            Qint(e).__iadd__(5)

    shared, _ = gate_count(add, [2, 8], qpu_size=11)
    independent, _ = gate_count(add, [2, 8])  # no free qubit for the AND
    assert shared["cnot"] < independent["cnot"]


@pytest.mark.parametrize("num_controls", [2, 3])
@pytest.mark.parametrize("gates", [(X, X), (Z, P(0.4)), (H, RX(0.3)), (RY(0.2), RZ(0.7))])
def test_shared_control_two_gates_reduce_cnots(num_controls, gates):
    """With two or three controls, any two base gates share the controls with
    fewer CNOTs (Theorem 4 and its corollary)."""

    def body(c, t):
        with control(c):
            for gate, qubit in zip(gates, t):
                gate(qubit)

    free = num_controls + 2 + 2
    shared, _ = gate_count(body, [num_controls, 2], qpu_size=free)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("ket._shared_control.MAX_SHARED_CONTROLS", 0)
        independent, _ = gate_count(body, [num_controls, 2], qpu_size=free)
    assert shared["cnot"] < independent["cnot"]


def test_four_controls_are_not_shared():
    """Libket has no relative-phase C4X: the controls are added to each gate."""

    def body(c, t):
        with control(c):
            P(0.4, t[0])
            P(0.4, t[1])

    counts, _ = gate_count(body, [4, 2], qpu_size=10)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("ket._shared_control.MAX_SHARED_CONTROLS", 0)
        independent, _ = gate_count(body, [4, 2], qpu_size=10)
    assert counts["cnot"] == independent["cnot"]


def test_shared_control_inside_around_compute():
    """The AND qubit is not blocked by an enclosing ``around`` (QUBO oracle)."""

    def load(controls, energy):
        with control(controls):
            energy += 3

    def circuit(q):
        energy = Qint(q.ket_process.alloc_aux(3))
        with around(load, q[0] + q[1], energy):
            Z(energy[0])
        return energy

    assert _equal_up_to_phase(*_states(circuit, 2, extra=4, kbw_extra=3))


def test_live_split_shared_control():
    """Two shared-control blocks separated by a dump: the uncompute of the
    first AND (already executed) must not cancel with the compute of the
    second one."""

    def circuit(q):
        for angle in (0.3, 0.5):
            with control(q[0] + q[1]):
                RX(angle, q[2])
                RY(angle, q[3])
            QuantumState(q).get()
            Z(q[3])

    assert _equal_up_to_phase(*_states(circuit, 4))


def test_live_repeated_dump_keeps_executed_gates():
    """Flattening the circuit again after a dump must not simplify the
    executed gates differently (a gate would be executed twice or skipped)."""

    def circuit(q):
        register = Qint(q[0:3])  # the QFT is a template block
        with around(H, q[3]):
            ctrl(q[3], register.__iadd__)(6)
        with around(ctrl(q[1] + q[3], X), q[4]):
            ctrl(q[4], P(1.1))(q[0])
        QuantumState(q).get()  # the comparison dumps again

    assert _equal_up_to_phase(*_states(circuit, 6))


def test_identity_on_zero_is_exact():
    """Controlled gates on qubits in |0> are skipped; the result is unchanged."""

    def circuit(q):
        fresh = q.ket_process.alloc(3)
        ctrl(fresh[0], X)(q[0])  # control in |0>: identity
        ctrl(q[1], P(0.7))(fresh[1])  # phase on a target in |0>: identity
        H(fresh[2])
        ctrl(fresh[2], RX(0.8))(q[2])  # control no longer in |0>
        return fresh

    assert _equal_up_to_phase(*_states(circuit, 3, extra=3, kbw_extra=3))


def test_gate_count_inputs_are_unknown():
    counts, _ = gate_count(lambda c, t: ctrl(c, X)(t), [2, 1])
    assert counts["cnot"] == 6
