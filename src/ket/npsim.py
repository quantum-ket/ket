"""Live Execution Simulator Based on NumPy

This simulator is available as a demonstration of the LiveExecution class for
constructing live execution targets using Python. It is not intended to be a
high-performance simulator. For better performance, use the KBW simulator.
"""

# SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

from random import choices
from .clib.libket.execution import LiveExecution

try:
    import numpy as np

    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False


class NPSim(LiveExecution):
    """NumPy based Live Execution simulator.

    Args:
        num_qubits: number of qubits.
    """

    def __init__(self, num_qubits):
        if not NP_AVAILABLE:
            raise RuntimeError(
                "NumPy is not available. Please install it with: pip install numpy"
            )

        super().__init__()

        self.num_qubit = num_qubits
        self.state_vector = np.zeros((1 << num_qubits, 1), dtype=complex)
        self.state_vector[0, 0] = 1.0

    def connect(self):
        return self.configure(self.num_qubit, True)

    def _apply_gate(self, gate, target):
        gate_kron = np.kron(
            np.kron(np.eye(1 << target, 1 << target, dtype=complex), gate),
            np.eye(
                1 << (self.num_qubit - target - 1),
                1 << (self.num_qubit - target - 1),
                dtype=complex,
            ),
        )
        self.state_vector = gate_kron @ self.state_vector

    def pauli_x(self, target, control):
        assert len(control) <= 1, "Multi-control qubits are not supported"
        gate = np.array([[0, 1], [1, 0]], dtype=complex)
        if len(control) == 0:
            self._apply_gate(gate, target)
        else:
            gate_mat = np.zeros(
                (1 << self.num_qubit, 1 << self.num_qubit), dtype=complex
            )
            for col in map(
                lambda s: list(f"{s:0{self.num_qubit}b}"), range(1 << self.num_qubit)
            ):
                lin = col[:]
                if col[control[0]] == "1":
                    lin[target] = "0" if col[target] == "1" else "1"
                gate_mat[int("".join(lin), 2), int("".join(col), 2)] = 1.0
            self.state_vector = gate_mat @ self.state_vector

    def pauli_y(self, target, control):
        assert len(control) == 0, "control qubits are not supported"
        gate = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self._apply_gate(gate, target)

    def pauli_z(self, target, control):
        assert len(control) == 0, "control qubits are not supported"
        gate = np.array([[1, 0], [0, -1]], dtype=complex)
        self._apply_gate(gate, target)

    def hadamard(self, target, control):
        assert len(control) == 0, "Control qubits are not supported"
        gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        self._apply_gate(gate, target)

    def rotation_x(self, target, control, angle):
        assert len(control) == 0, "Control qubits are not supported"
        gate = np.array(
            [
                [np.cos(angle / 2), -1j * np.sin(angle / 2)],
                [-1j * np.sin(angle / 2), np.cos(angle / 2)],
            ],
            dtype=complex,
        )
        self._apply_gate(gate, target)

    def rotation_y(self, target, control, angle):
        assert len(control) == 0, "Control qubits are not supported"
        gate = np.array(
            [
                [np.cos(angle / 2), -np.sin(angle / 2)],
                [np.sin(angle / 2), np.cos(angle / 2)],
            ],
            dtype=complex,
        )
        self._apply_gate(gate, target)

    def rotation_z(self, target, control, angle):
        assert len(control) == 0, "Control qubits are not supported"
        gate = np.array(
            [[np.exp(-1j * angle / 2), 0], [0, np.exp(1j * angle / 2)]], dtype=complex
        )
        self._apply_gate(gate, target)

    def phase(self, target, control, angle):
        assert len(control) == 0, "Control qubits are not supported"
        gate = np.array([[1, 0], [0, np.exp(1j * angle)]], dtype=complex)
        self._apply_gate(gate, target)

    def _p1(self, qubit: int) -> float:
        shift = self.num_qubit - qubit - 1
        mask = 1 << shift
        return sum(
            np.abs(amp[0]) ** 2 if (state & mask) != 0 else 0.0
            for state, amp in enumerate(self.state_vector)
        )

    def _colapse(self, qubit, m, p):
        shift = self.num_qubit - qubit - 1
        mask = 1 << shift
        target_bit = m << shift
        for state, amp in enumerate(self.state_vector):
            if (state & mask) == target_bit:
                self.state_vector[state, 0] = amp[0] / np.sqrt(p)
            else:
                self.state_vector[state, 0] = 0.0

    def measure(self, qubits: list[int]) -> int:
        result = []
        for q in reversed(qubits):
            p1 = self._p1(q)
            p0 = 1.0 - p1
            m, p = choices([(0, p0), (1, p1)], [p0, p1])[0]
            result.append(m)
            self._colapse(q, m, p)
        return int("".join(map(str, result)), 2)

    def dump(self, qubits: list[int]) -> dict:
        basis_states = []
        amplitudes_real = []
        amplitudes_imag = []

        for state_idx, amp in enumerate(self.state_vector):
            scalar_amp = amp[0]
            if np.abs(scalar_amp) <= 1e-10:
                continue

            bin_str = f"{state_idx:0{self.num_qubit}b}"
            extracted_state = int("".join(bin_str[q] for q in qubits), 2)

            basis_states.append([extracted_state])
            amplitudes_real.append(float(np.real(scalar_amp)))
            amplitudes_imag.append(float(np.imag(scalar_amp)))

        return {
            "basis_states": basis_states,
            "amplitudes_real": amplitudes_real,
            "amplitudes_imag": amplitudes_imag,
        }
