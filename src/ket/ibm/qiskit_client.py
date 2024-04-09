from math import pi
from typing import Any

try:
    from qiskit import QuantumCircuit
    from qiskit.providers import Backend
    from qiskit.circuit import library, Gate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, Session
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
except ImportError:
    raise ImportError(
        "QiskitClient requires the qiskit module to be used. You can install them"
        "alongside ket by running `pip install ket[ibm]`."
    )


class QiskitClient:
    def __init__(
        self, num_qubits: int, backend: Backend, service: QiskitRuntimeService | None
    ) -> None:
        self.num_qubits = num_qubits
        self.backend = backend
        self.service = service
        self.quantum_circuit = QuantumCircuit(num_qubits, num_qubits)

        self._measurement_map = {}
        self._sample_map = {}
        self._samples = {}
        self._exp_values = []

    def process_instructions(
        self, instructions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self._send_instructions(instructions)
        formatted_result = self._format_result()
        return formatted_result

    def _send_instructions(self, instructions: list[dict[str, Any]]):
        qubit_map = list(range(self.num_qubits))
        qubit_stack = list(range(self.num_qubits))
        observables = []

        for inst in instructions:
            if "Alloc" in inst:
                qubit_map[inst["Alloc"]["target"]] = qubit_stack.pop()

            elif "Free" in inst:
                qubit_stack.append(qubit_map[inst["Free"]["target"]])

            elif "Gate" in inst:
                gate_type = inst["Gate"]["gate"]
                gate: Gate = self._get_gate(gate_type)

                control = [qubit_map[qubit] for qubit in inst["Gate"]["control"]]
                if len(control):
                    gate = gate.control(len(control))
                self.quantum_circuit.append(
                    gate, control + [qubit_map[inst["Gate"]["target"]]]
                )

            elif "Measure" in inst:
                qubits = [qubit_map[qubit] for qubit in inst["Measure"]["qubits"]]
                self._measurement_map[inst["Measure"]["output"]] = qubits
                self.quantum_circuit.measure(qubits, qubits)

            elif "ExpValue" in inst:
                hamiltonian: dict = inst["ExpValue"]["hamiltonian"]
                observables.append(self._build_observable(hamiltonian))

            elif "Sample" in inst:
                qubits = [qubit_map[qubit] for qubit in inst["Sample"]["qubits"]]
                self._sample_map[inst["Sample"]["output"]] = qubits
                self.quantum_circuit.measure(qubits, qubits)

            elif "Dump" in inst:
                raise RuntimeError("Operation not supported")

            else:
                raise RuntimeError("Unknown operation")

        pm = generate_preset_pass_manager(
            target=self.backend.target, optimization_level=1
        )
        isa_circuit = pm.run(self.quantum_circuit)

        with Session(service=self.service, backend=self.backend) as session:
            if self._sample_map or self._measurement_map:
                sampler = Sampler(session=session)
                result_data = sampler.run([isa_circuit]).result()[0].data
                result = (
                    result_data.c.get_counts()
                    if hasattr(result_data, "c")
                    else result_data.cr.get_counts()
                )
                self._samples = result

            for observable in observables:
                isa_observable = observable.apply_layout(isa_circuit.layout)
                estimator = Estimator(session=session)
                job = estimator.run([(isa_circuit, isa_observable)])
                pub_result = job.result()[0].data.evs

                result = (
                    pub_result.tolist()
                    if type(pub_result.tolist()) == list
                    else [float(pub_result)]
                )
                self._exp_values.extend(result)

    def _format_result(self) -> dict[str, Any]:
        result_dict = {
            "measurements": [0] * len(self._measurement_map),
            "exp_values": [],
            "samples": [0] * len(self._sample_map),
            "dumps": [],
            "execution_time": None,
        }

        if self._measurement_map:
            meas_bin = max(self._samples, key=self._samples.get)
            reversed_meas_bin = meas_bin[::-1]

            # meas_index is the index of the measurement in the measurement map, as
            # there can be multiple measurements in a single instruction.
            # qubits is the list of qubits that were measured in the measurement.
            for meas_index, qubits in self._measurement_map.items():
                result_dict["measurements"][meas_index] = int(
                    "".join([reversed_meas_bin[qubit] for qubit in qubits]), 2
                )

        if self._sample_map:
            for k, v in self._sample_map.items():
                result_dict["samples"][k] = (
                    [
                        int("".join([measurement[::-1][i] for i in v]), 2)
                        for measurement in self._samples.keys()
                    ],
                    list(self._samples.values()),
                )

        result_dict["exp_values"] = self._exp_values
        return result_dict

    def _get_gate(self, gate_type) -> Gate:
        if "PauliX" in gate_type:
            return library.XGate()
        elif "PauliY" in gate_type:
            return library.YGate()
        elif "PauliZ" in gate_type:
            return library.ZGate()
        elif "Hadamard" in gate_type:
            return library.HGate()
        elif "Rotation" in gate_type or "Phase" in gate_type:
            """gate_type here is a dict"""
            rotation_type: str = gate_type.keys()[0]
            rot = gate_type[rotation_type]
            theta = 0
            if "Scalar" in rot:
                theta = rot["Scalar"]
            else:
                pi_fraction = rot["PiFraction"]
                theta = pi * pi_fraction["top"] / pi_fraction["bottom"]
            return self._get_rotation_or_phase_gate(gate_type, theta)
        else:
            raise RuntimeError("Unknown gate")

    def _get_rotation_or_phase_gate(self, gate_type, theta) -> Gate:
        if "RotationX" in gate_type:
            return library.RXGate(theta)
        elif "RotationY" in gate_type:
            return library.RYGate(theta)
        elif "RotationZ" in gate_type:
            return library.RZGate(theta)
        elif "Phase" in gate_type:
            return library.U1Gate(theta)

    def _build_observable(self, hamiltonian: dict[str, Any]) -> SparsePauliOp:
        """Iterates over each dict term in ket's hamiltonian, transforming them into
        n_qubits long string terms, combining Pauli operators when viable, to form a
        qiskt hamiltonian string.

        Example:
        hamiltonian = {
            "coefficients": [1, 1],
            "products": [
                [{"qubit": 0, "pauli": "PauliX"}],
                [{"qubit": 0, "pauli": "PauliY"}],
            ],
        }
        qiskit_hamiltonian = ["XII", "YII"]
        qiskit_coeffs = [1, 1]

        The qiskit_hamiltonian and qiskit_coeffs are then used to create a SparsePauliOp
        object, which is then used to calculate the expectation value.
        """
        hamiltonian_sum: list[list[dict]] = hamiltonian["products"]
        qiskit_hamiltonian = []
        qiskit_coeffs = []

        for coef_index, product in enumerate(hamiltonian_sum):
            qiskit_hmlt_product = []
            i = 0
            while i < len(product):
                qiskit_hamlt_term = ["I"] * self.num_qubits
                while i < len(product):
                    current_qubit: int = product[i]["qubit"]
                    if qiskit_hamlt_term[current_qubit] != "I":
                        break
                    pauli_type: str = product[i]["pauli"][-1]
                    qiskit_hamlt_term[current_qubit] = pauli_type
                    i += 1
                qiskit_hmlt_product.append("".join(qiskit_hamlt_term))

            qiskit_hamiltonian.extend(qiskit_hmlt_product)
            qiskit_coeffs.extend(
                [hamiltonian["coefficients"][coef_index]] * len(qiskit_hmlt_product)
            )

        observable = SparsePauliOp(qiskit_hamiltonian, qiskit_coeffs)
        return observable
