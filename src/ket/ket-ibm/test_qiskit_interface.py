from qiskit_interface import QiskitInterface
from qiskit_aer import AerSimulator
import ket

def test_measure():
    num_qubits = 2
    interface = QiskitInterface(num_qubits, AerSimulator())
    process = ket.Process(interface.make_configuration())
    a, b = process.alloc(2)

    ket.H(a)
    ket.CNOT(a, b)
    ket.measure(a + b)

    process.execute()
    assert True

def test_sample():
    num_qubits = 2
    interface = QiskitInterface(num_qubits, AerSimulator())
    process = ket.Process(interface.make_configuration())
    a, b = process.alloc(2)

    ket.H(a)
    ket.CNOT(a, b)
    ket.sample(a)

    process.execute()
    assert True


def test_exp_value():
    num_qubits = 2
    interface = QiskitInterface(num_qubits, AerSimulator())
    process = ket.Process(interface.make_configuration())
    a, b = process.alloc(2)

    ket.CNOT(ket.H(a), b)
    ket.Y(a)
    ket.exp_value(ket.Pauli("X", a + b, coef=0.5))

    process.execute()
    assert True
