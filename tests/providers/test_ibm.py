# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2026 Otávio Augusto de Santana Jatobá <otavio.jatoba@grad.ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import math
import ket

# Gracefully skip these tests if Qiskit is not installed
qiskit = pytest.importorskip("qiskit", reason="Qiskit SDK is not installed")
from qiskit_aer import AerSimulator
from ket.ibm import IBMDevice

BACKEND = AerSimulator()
NUM_QUBITS = 3


def test_ibm_sample():
    """Test the sampling operation dispatch using the IBM Qiskit backend."""
    device = IBMDevice(backend=BACKEND)
    ket_process = ket.Process(execution_target=device)

    q = ket_process.alloc(NUM_QUBITS)
    ket.CNOT(ket.H(q[0]), q[1])
    ket.sample(q)

    ket_process.execute()


def test_ibm_exp_value():
    """Test the expectation value operation dispatch using the IBM Qiskit backend."""

    # 1. Execute purely in Ket native simulator for the exact baseline
    ket_process = ket.Process(execution="batch")
    q = ket_process.alloc(NUM_QUBITS)
    ket.CNOT(ket.H(q[0]), q[1])
    hamiltonian = ket.Pauli("Z", q)

    native_ev = ket.exp_value(hamiltonian)

    # 2. Execute via IBMDevice using EstimatorV2
    device = IBMDevice(backend=BACKEND)
    ibm_process = ket.Process(execution_target=device)
    t = ibm_process.alloc(NUM_QUBITS)
    ket.CNOT(ket.H(t[0]), t[1])
    hamiltonian_ibm = ket.Pauli("Z", t)

    ibm_ev = ket.exp_value(hamiltonian_ibm)

    # Ensure the results from Qiskit and the Ket native simulator match
    assert math.isclose(native_ev.get(), ibm_ev.get(), rel_tol=1e-5, abs_tol=1e-5)


def test_ibm_sample_counts_add_up():
    """All shots must be counted when the device has more qubits than sampled."""
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    shots = 1024
    device = IBMDevice(backend=AerSimulator.from_backend(FakeManilaV2()))
    ket_process = ket.Process(execution_target=device)

    q = ket_process.alloc(2)
    ket.CNOT(ket.H(q[0]), q[1])
    result = ket.sample(q, shots)

    assert sum(result.get().values()) == shots
