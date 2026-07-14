"""Module providing functionality to interact with IBM Quantum and IBM Cloud devices.

Note:
    This module requires additional dependencies from ``ket-lang[ibm]``.

    Install with: ``pip install ket-lang[ibm]``.

"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2024 Otávio Augusto de Santana Jatobá
# <otavio.jatoba@grad.ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from .clib.libket.execution import BatchExecution

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import library
    from qiskit.providers import BackendV2
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2 as Sampler, EstimatorV2 as Estimator
    from qiskit_aer import AerSimulator

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class IBMDevice(BatchExecution):  # pylint: disable=too-many-instance-attributes
    """IBM Qiskit backend for Ket process.

    Args:
        backend: The backend to be used for the quantum execution. If not
            provided, it defaults to the AerSimulator.
        optimization_level: The optimization level for the pass manager.
    """

    def __init__(
        self,
        backend: BackendV2 | None = None,
        num_qubits: int | None = None,
        gradient: bool = False,
        optimization_level: int = 2,
        qiskit_compiler=True,
    ):
        if not QISKIT_AVAILABLE:
            raise ImportError(
                "IBMDevice requires the qiskit module to be used. You can install them"
                " alongside ket by running `pip install ket-lang[ibm]`."
            )

        super().__init__()

        if backend is None:
            backend = AerSimulator()

        self.num_qubits = backend.configuration().n_qubits
        if num_qubits is not None and num_qubits < self.num_qubits:
            self.num_qubits = num_qubits

        self.backend = backend
        self.decompose = not qiskit_compiler

        self.pm = generate_preset_pass_manager(
            backend=self.backend,
            optimization_level=optimization_level,
        )

        self.gradient = gradient

    @staticmethod
    def _get_gate(gate):
        match gate:
            case "Hadamard":
                return library.HGate()
            case "PauliX":
                return library.XGate()
            case "PauliY":
                return library.YGate()
            case "PauliZ":
                return library.ZGate()
            case {"RotationX": {"Value": angle}}:
                return library.RXGate(angle)
            case {
                "RotationX": {
                    "Ref": {
                        "index": _,
                        "multiplier": multiplier,
                        "value": value,
                    }
                }
            }:
                return library.RXGate(multiplier * value)
            case {"RotationY": {"Value": angle}}:
                return library.RYGate(angle)
            case {
                "RotationY": {
                    "Ref": {
                        "index": _,
                        "multiplier": multiplier,
                        "value": value,
                    }
                }
            }:
                return library.RYGate(multiplier * value)
            case {"RotationZ": {"Value": angle}}:
                return library.RZGate(angle)
            case {
                "RotationZ": {
                    "Ref": {
                        "index": _,
                        "multiplier": multiplier,
                        "value": value,
                    }
                }
            }:
                return library.RZGate(multiplier * value)
            case {"Phase": {"Value": angle}}:
                return library.PhaseGate(angle)
            case {
                "Phase": {
                    "Ref": {
                        "index": _,
                        "multiplier": multiplier,
                        "value": value,
                    }
                }
            }:
                return library.PhaseGate(multiplier * value)
            case _:
                raise RuntimeError(f"Undefined gate '{gate}'")

    def _build_circuit(self, gates):
        """Build a Qiskit QuantumCircuit from a NativeGate list."""
        circuit = QuantumCircuit(self.num_qubits)

        for gate_inst in gates:
            gate = gate_inst["gate"]
            target = gate_inst["target"]
            control = gate_inst["control"]
            decomposed = gate_inst["decomposed"]

            if decomposed is not None:
                for gate in decomposed:
                    match gate:
                        case {"U": (gate, target)}:
                            circuit.append(self._get_gate(gate), [target])
                        case {"CNOT": (control, target)}:
                            circuit.cx(control, target)
            else:
                gate = self._get_gate(gate)
                if control:
                    circuit.append(gate.control(len(control)), control + [target])
                else:
                    circuit.append(gate, [target])

        return circuit

    def sample(self, gates, qubits_to_sample, shots):
        """Execute the circuit and sample the given qubits.

        .. warning::
            This method is called by Libket and should not be called directly.
        """
        circuit = self._build_circuit(gates)
        circuit.measure_all(inplace=True)

        isa_circuit = self.pm.run(circuit)

        sampler = Sampler(mode=self.backend)
        job = sampler.run([isa_circuit], shots=shots)
        raw_counts = job.result()[0].data["meas"].get_counts()

        parsed_counts = {
            int("".join(s[-q - 1] for q in qubits_to_sample), 2): c
            for s, c in raw_counts.items()
        }

        mask_64bit = (1 << 64) - 1
        num_chunks = (len(qubits_to_sample) + 63) // 64

        states, counts = zip(*parsed_counts.items())
        return [
            [(s >> (64 * i)) & mask_64bit for i in range(num_chunks)] for s in states
        ], list(counts)

    def exp_value(self, gates, hamiltonian_list):
        """Execute the circuit and compute expectation values.

        .. warning::
            This method is called by Libket and should not be called directly.
        """
        circuit = self._build_circuit(gates)
        observables = []

        for hamiltonian in hamiltonian_list:
            pauli_strings = []
            for pauli, coef in zip(
                hamiltonian["pauli_strings"], hamiltonian["coefficients"]
            ):
                string = ["I"] * self.num_qubits
                for item in pauli:
                    string[item["qubit"]] = item["pauli"][-1]
                pauli_strings.append(("".join(reversed(string)), coef))

            observables.append(SparsePauliOp.from_list(pauli_strings))

        isa_circuit = self.pm.run(circuit)
        isa_observables = [o.apply_layout(isa_circuit.layout) for o in observables]
        estimator = Estimator(mode=self.backend)
        job = estimator.run([(isa_circuit, o) for o in isa_observables])
        return [float(result.data.evs) for result in job.result()]

    def connect(self):
        """Connect to the device and return the configuration."""
        return self.configure(
            self.num_qubits,
            gradient=self.gradient,
            decompose=self.decompose,
        )
