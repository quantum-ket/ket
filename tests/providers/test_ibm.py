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
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from ket.ibm import IBMDevice

BACKEND = AerSimulator()
NUM_QUBITS = 3


def _build_qiskit_bell_circuit():
    """Helper to build a native Qiskit circuit for comparison."""
    qiskit_circuit = qiskit.QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
    qiskit_circuit.h(0)
    qiskit_circuit.cx(0, 1)
    return qiskit_circuit


def test_ibm_measure():
    """Test the measurement operation dispatch using the IBM Qiskit backend."""
    device = IBMDevice(backend=BACKEND)
    # Correctly bind the execution target using the Ket Process API
    ket_process = ket.Process(execution_target=device)

    q = ket_process.alloc(NUM_QUBITS)
    ket.CNOT(ket.H(q[0]), q[1])
    ket.sample(q)

    ket_process.execute()

    # Compare with native Qiskit circuit compilation
    qubits = list(range(NUM_QUBITS))
    qiskit_circuit = _build_qiskit_bell_circuit()
    qiskit_circuit.measure(qubits, qubits)

    pm = generate_preset_pass_manager(backend=BACKEND, optimization_level=1)
    isa_qc = pm.run(qiskit_circuit)

    # Ensure all instructions mapped by IBMDevice match the Qiskit ISA
    for interface_inst in device.circuit.data:
        assert interface_inst in isa_qc.data


def test_ibm_sample():
    """Test the sampling operation dispatch using the IBM Qiskit backend."""
    device = IBMDevice(backend=BACKEND)
    ket_process = ket.Process(execution_target=device)

    q = ket_process.alloc(NUM_QUBITS)
    ket.CNOT(ket.H(q[0]), q[1])
    ket.sample(q)

    ket_process.execute()

    # Compare with native Qiskit circuit compilation
    qubits = list(range(NUM_QUBITS))
    qiskit_circuit = _build_qiskit_bell_circuit()
    qiskit_circuit.measure(qubits, qubits)

    pm = generate_preset_pass_manager(backend=BACKEND, optimization_level=1)
    isa_qc = pm.run(qiskit_circuit)

    for interface_inst in device.circuit.data:
        assert interface_inst in isa_qc.data


def test_ibm_exp_value():
    """Test the expectation value operation dispatch using the IBM Qiskit backend."""

    # 1. Execute purely in Ket native simulator for the exact baseline
    ket_process = ket.Process()
    q = ket_process.alloc(NUM_QUBITS)
    ket.CNOT(ket.H(q[0]), q[1])
    hamiltonian = ket.Pauli("Z", q)

    native_ev = ket.exp_value(hamiltonian)
    ket_process.execute()

    # 2. Execute via IBMDevice using EstimatorV2
    device = IBMDevice(backend=BACKEND)
    ibm_process = ket.Process(execution_target=device)
    t = ibm_process.alloc(NUM_QUBITS)
    ket.CNOT(ket.H(t[0]), t[1])
    hamiltonian_ibm = ket.Pauli("Z", t)

    ibm_ev = ket.exp_value(hamiltonian_ibm)
    ibm_process.execute()

    # Ensure the results from Qiskit and the Ket native simulator match
    assert math.isclose(native_ev.value, ibm_ev.value, rel_tol=1e-5, abs_tol=1e-5)
