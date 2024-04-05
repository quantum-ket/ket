import math
import ket

import qiskit
from qiskit_aer import AerSimulator
from ket.ibm.qiskit_client import QiskitClient


def test_circuit():
    backend = AerSimulator()
    ket_process = ket.Process()
    q = ket_process.alloc(3)
    ket.CNOT(ket.H(q[0]), q[1])
    ket.CNOT(ket.H(q[1]), q[2])
    ket.sample(q)

    instructions = ket_process.get_instructions()
    client = QiskitClient(backend, num_qubits=3)
    client.process_instructions(instructions)

    qiskit_circuit = qiskit.QuantumCircuit(3, 3)
    qiskit_circuit.h(2)
    qiskit_circuit.cx(2, 1)
    qiskit_circuit.h(1)
    qiskit_circuit.cx(1, 0)
    qiskit_circuit.measure([0, 1, 2], [0, 1, 2])
    transpiled_circuit = qiskit.transpile(qiskit_circuit, backend)
    backend.run(transpiled_circuit).result()

    for interface_inst in client.quantum_circuit.data:
        assert interface_inst in transpiled_circuit.data


def test_exp_value():
    backend = AerSimulator()

    ket_process = ket.Process()
    q = ket_process.alloc(2)
    ket.CNOT(ket.H(q[0]), q[1])
    hamiltonian = (ket.Pauli("X", q) * ket.Pauli("Z", q) * ket.Pauli("Y", q)) + (
        2 * ket.Pauli("Y", q) * ket.Pauli("X", q)
    )
    ket_result = ket.exp_value(hamiltonian)
    final_ket_result = float(ket_result.value)

    instructions = ket_process.get_instructions()
    client = QiskitClient(backend, num_qubits=2)
    client_result = client.process_instructions(instructions)
    final_client_result = float(client_result["exp_values"][0])

    assert math.isclose(final_ket_result, final_client_result, rel_tol=1e-9)
