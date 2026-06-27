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

        self.pm = generate_preset_pass_manager(
            backend=self.backend,
            optimization_level=optimization_level,
        )

        self.gradient = gradient

    def _build_circuit(self, gates):
        """Build a Qiskit QuantumCircuit from a NativeGate list."""
        circuit = QuantumCircuit(self.num_qubits)

        for gate_name, params, qubits in gates:
            match gate_name:
                case "h":
                    g = library.HGate()
                case "x":
                    g = library.XGate()
                case "y":
                    g = library.YGate()
                case "z":
                    g = library.ZGate()
                case "rx":
                    g = library.RXGate(params[0])
                case "ry":
                    g = library.RYGate(params[0])
                case "rz":
                    g = library.RZGate(params[0])
                case "p":
                    g = library.PhaseGate(params[0])
                case "cnot":
                    g = library.CXGate()
                case "cz":
                    g = library.CZGate()
                case _:
                    raise RuntimeError(f"Undefined gate '{gate_name}'")

            circuit.append(g, qubits)

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
        return self.configure(gradient=self.gradient)
