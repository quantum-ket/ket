""" Test module for the IBMDevice class """

# SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2024 Otávio Augusto de Santana Jatobá
# <otavio.jatoba@grad.ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

try:
    import qiskit
    from qiskit_aer import AerSimulator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
except ImportError as exc:
    raise ImportError(
        "IBMDevice requires the qiskit module to be used. You can install them"
        "alongside ket by running `pip install ket[ibm]`."
    ) from exc

import math
import ket
from ket.ibm import IBMDevice


BACKEND = AerSimulator()
NUM_QUBITS = 3


class TestIBMDevice:
    """Test class for the IBMDevice use cases."""

    def _build_qiskit_bell_circuit(self):
        qiskit_circuit = qiskit.QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
        qiskit_circuit.h(0)
        qiskit_circuit.cx(0, 1)
        return qiskit_circuit

    def test_measure(self):
        """Tests the measurement operation using the IBMDevice"""

        device = IBMDevice(backend=BACKEND, num_qubits=NUM_QUBITS)
        ket_process = ket.Process(device.make_configuration())
        q = ket_process.alloc(NUM_QUBITS)
        ket.CNOT(ket.H(q[0]), q[1])
        ket.measure(q)
        ket_process.execute()

        qubits = list(range(NUM_QUBITS))
        qiskit_circuit = self._build_qiskit_bell_circuit()
        qiskit_circuit.measure(qubits, qubits)

        pm = generate_preset_pass_manager(backend=BACKEND, optimization_level=1)
        isa_qc = pm.run(qiskit_circuit)

        for interface_inst in device.circuit.data:
            assert interface_inst in isa_qc.data

        assert True
        print("Measurement test passed!")

    def test_sample(self):
        """Tests the sample operation using the IBMDevice"""

        device = IBMDevice(backend=BACKEND, num_qubits=NUM_QUBITS)
        ket_process = ket.Process(device.make_configuration())
        q = ket_process.alloc(NUM_QUBITS)
        ket.CNOT(ket.H(q[0]), q[1])
        ket.sample(q)
        ket_process.execute()

        qubits = list(range(NUM_QUBITS))
        qiskit_circuit = self._build_qiskit_bell_circuit()
        qiskit_circuit.measure(qubits, qubits)

        pm = generate_preset_pass_manager(backend=BACKEND, optimization_level=1)
        isa_qc = pm.run(qiskit_circuit)

        for interface_inst in device.circuit.data:
            assert interface_inst in isa_qc.data

        assert True
        print("Sample test passed!")

    def test_exp_value(self):
        """Tests the expectation value operation using the IBMDevice"""

        ket_process = ket.Process()
        q = ket_process.alloc(NUM_QUBITS)
        ket.CNOT(ket.H(q[0]), q[1])
        hamiltonian = ket.Pauli("Z", q)

        device = IBMDevice(backend=BACKEND, num_qubits=NUM_QUBITS)
        ibm_process = ket.Process(device.make_configuration())
        t = ibm_process.alloc(NUM_QUBITS)
        ket.CNOT(ket.H(t[0]), t[1])

        ket_result = ket.exp_value(hamiltonian).value
        device_result = ket.exp_value(hamiltonian).value

        ket_process.execute()
        ibm_process.execute()

        assert math.isclose(ket_result, device_result, rel_tol=1e-9)
        print("Expectation value test passed!")
