from __future__ import annotations
from .clib.libket import BatchExecution

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import library
    from qiskit.providers import BackendV2

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class IBMDevice(BatchExecution):
    """IBM Qiskit backend for Ket process.

    Args:
        backend: The backend to be used for the quantum execution.
        use_qiskit_transpiler: Use Qiskit transpiler instead of Ket's.
    """

    def __init__(
        self,
        backend: BackendV2,
        *,
        shots: int | None = None,
        classical_shadows: dict | None = None,
        use_qiskit_transpiler: bool = False,
    ):
        if not QISKIT_AVAILABLE:
            raise ImportError(
                "IBMDevice requires the qiskit module to be used. You can install them"
                " alongside ket by running `pip install ket[ibm]`."
            )

        if shots is not None and classical_shadows is not None:
            raise ValueError(
                "You cannot specify both 'shots' and 'classical_shadows'. "
                "Please choose one of them."
            )

        super().__init__()

        self.num_qubits = backend.configuration().n_qubits

        self.client = None
        self.backend = backend

        if not use_qiskit_transpiler:
            self.coupling_graph = (
                list(backend.coupling_map.graph.edge_list())
                if backend.coupling_map
                else [
                    [i, j]
                    for i in range(self.num_qubits)
                    for j in range(self.num_qubits)
                    if i != j
                ]
            )
        else:
            self.coupling_graph = None

        self.circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
        self.parameters = None

        self.shots = 2048 if shots is None and classical_shadows is None else shots
        self.classical_shadows = classical_shadows

        self.qubits_from_sample = None
        self.result = None

    def clear(self):
        self.circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
        self.parameters = None
        self.qubits_from_sample = None
        self.result = None

    def submit_execution(self, circuit, parameters):
        self.parameters = parameters
        self.process_instructions(circuit)

    def get_result(self):

        self.result = {
            "".join(s[-q - 1] for q in self.qubits_from_sample): c
            for s, c in self.result.items()
        }

        result = list(zip(*((int(s, 2), c) for s, c in self.result.items())))

        results_dict = {
            "measurements": [],
            "exp_values": [],
            "samples": [result],
            "dumps": [],
            "gradients": None,
        }

        return results_dict

    def pauli_x(self, target, control):
        gate = library.XGate()
        if len(control) >= 1:
            gate = gate.control(len(control))
        self.circuit.append(gate, control + [target])

    def pauli_y(self, target, control):
        gate = library.YGate()
        if len(control) >= 1:
            gate = gate.control(len(control))
        self.circuit.append(gate, control + [target])

    def pauli_z(self, target, control):
        gate = library.ZGate()
        if len(control) >= 1:
            gate = gate.control(len(control))
        self.circuit.append(gate, control + [target])

    def hadamard(self, target, control):
        gate = library.HGate()
        if len(control) >= 1:
            gate = gate.control(len(control))
        self.circuit.append(gate, control + [target])

    def rotation_x(self, target, control, **kwargs):
        gate = library.RXGate(kwargs["Value"])
        if len(control) >= 1:
            gate = gate.control(len(control))
        self.circuit.append(gate, control + [target])

    def rotation_y(self, target, control, **kwargs):
        gate = library.RYGate(
            kwargs["Value"]
            if "Value" in kwargs
            else self.parameters[kwargs["Ref"]["index"]] * kwargs["Ref"]["multiplier"]
        )
        if len(control) >= 1:
            gate = gate.control(len(control))
        self.circuit.append(gate, control + [target])

    def rotation_z(self, target, control, **kwargs):
        gate = library.RZGate(
            kwargs["Value"]
            if "Value" in kwargs
            else self.parameters[kwargs["Ref"]["index"]] * kwargs["Ref"]["multiplier"]
        )
        if len(control) >= 1:
            gate = gate.control(len(control))
        self.circuit.append(gate, control + [target])

    def phase(self, target, control, **kwargs):
        gate = library.PhaseGate(
            kwargs["Value"]
            if "Value" in kwargs
            else self.parameters[kwargs["Ref"]["index"]] * kwargs["Ref"]["multiplier"]
        )
        if len(control) >= 1:
            gate = gate.control(len(control))
        self.circuit.append(gate, control + [target])

    def sample(self, _, qubits, shots):
        self.circuit.measure(qubits, qubits)
        self.result = self.backend.run(self.circuit, shots=shots).result().get_counts()
        self.qubits_from_sample = qubits

    def connect(self):
        self.clear()

        qpu_params = {
            "coupling_graph": self.coupling_graph,
            "u4_gate_type": "CX",
            "u2_gate_set": "All",
        }

        return super().configure(
            num_qubits=self.num_qubits,
            direct_sample_exp_value=self.shots,
            classical_shadows_exp_value=self.classical_shadows,
            qpu=qpu_params if self.coupling_graph is not None else None,
        )
