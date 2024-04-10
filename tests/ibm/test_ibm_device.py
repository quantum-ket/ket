import ket
import math
import qiskit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from ket.ibm import IBMDevice

num_qubits = 3
aer_sim = AerSimulator()


def test_measure():
    device = IBMDevice(num_qubits=num_qubits, backend=aer_sim, service=None)
    ket_process = ket.Process(device.make_configuration())
    q = ket_process.alloc(3)
    ket.CNOT(ket.H(q[0]), q[1])
    ket.CNOT(ket.H(q[1]), q[2])
    ket.measure(q)
    ket_process.execute()

    qiskit_circuit = qiskit.QuantumCircuit(num_qubits, num_qubits)
    qiskit_circuit.h(2)
    qiskit_circuit.cx(2, 1)
    qiskit_circuit.h(1)
    qiskit_circuit.cx(1, 0)
    qiskit_circuit.measure([0, 1, 2], [0, 1, 2])

    pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
    isa_qc = pm.run(qiskit_circuit)

    for interface_inst in device.client.qiskit_builder.circuit.data:
        assert interface_inst in isa_qc.data

    assert True


def test_sample():
    device = IBMDevice(num_qubits=num_qubits, backend=aer_sim, service=None)
    ket_process = ket.Process(device.make_configuration())
    q = ket_process.alloc(3)
    ket.CNOT(ket.H(q[0]), q[1])
    ket.CNOT(ket.H(q[1]), q[2])
    ket.sample(q)
    ket_process.execute()

    qiskit_circuit = qiskit.QuantumCircuit(num_qubits, num_qubits)
    qiskit_circuit.h(2)
    qiskit_circuit.cx(2, 1)
    qiskit_circuit.h(1)
    qiskit_circuit.cx(1, 0)
    qiskit_circuit.measure([0, 1, 2], [0, 1, 2])

    pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
    isa_qc = pm.run(qiskit_circuit)

    for interface_inst in device.client.qiskit_builder.circuit.data:
        assert interface_inst in isa_qc.data

    assert True


def test_exp_value():
    ket_result = isolate_processing("ket")
    device_result = isolate_processing("device")
    print(f"ket_result: {ket_result}")
    print(f"device_result: {device_result}")

    assert math.isclose(ket_result, device_result, rel_tol=1e-9)


def isolate_processing(source):
    device = IBMDevice(num_qubits=num_qubits, backend=aer_sim, service=None)
    p = ket.Process() if source == "ket" else ket.Process(device.make_configuration())
    q = p.alloc(num_qubits)
    ket.CNOT(ket.H(q[0]), q[1])
    ket.CNOT(ket.H(q[1]), q[2])
    hamiltonian = (ket.Pauli("X", q) * ket.Pauli("Z", q) * ket.Pauli("Y", q)) + (
        2 * ket.Pauli("Y", q) * ket.Pauli("X", q)
    )
    return ket.exp_value(hamiltonian).value


if __name__ == "__main__":
    test_measure()
    test_sample()
    test_exp_value()
