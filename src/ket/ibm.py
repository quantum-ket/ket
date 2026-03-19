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

from .clib.libket import BatchExecution

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

    The arguments ``shots`` and ``classical_shadows`` control how the
    execution is performed for estimating expectation values of an
    Hamiltonian term. Only one of these arguments can be specified at a time.

    Args:
        backend: The backend to be used for the quantum execution. If not
            provided, it defaults to the AerSimulator.
    """

    def __init__(
        self,
        backend: BackendV2 | None = None,
        optimization_level=2,
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

        self.client = None
        self.backend = backend

        self.pm = generate_preset_pass_manager(
            backend=self.backend,
            optimization_level=optimization_level,
        )

        self.circuit = None
        self.parameters = None

        self.qubits_from_sample = None
        self.result = None
        self.exp_value_result = []

    def clear(self):
        self.circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
        self.parameters = None
        self.qubits_from_sample = None
        self.result = None

    def submit_execution(self, circuit, parameters):
        self.parameters = parameters
        self.process_instructions(circuit)

    def get_result(self):
        if self.result is not None:
            self.result = {
                "".join(s[-q - 1] for q in self.qubits_from_sample): c
                for s, c in self.result.items()
            }
            result = [list(zip(*((int(s, 2), c) for s, c in self.result.items())))]
        else:
            result = []

        results_dict = {
            "measurements": [],
            "exp_values": self.exp_value_result,
            "samples": result,
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

        isa_circuit = self.pm.run(self.circuit)
        sampler = Sampler(mode=self.backend)
        job = sampler.run([isa_circuit], shots=shots)
        self.result = job.result()[0].data.c.get_counts()
        self.qubits_from_sample = qubits

    def exp_value(self, _, hamiltonian):
        """Compute the expectation value.

        .. warning::
            This method is called by Libket and should not be called directly.
        """
        pauli_strings = []
        for pauli, coef in zip(hamiltonian["products"], hamiltonian["coefficients"]):
            string = ["I"] * self.num_qubits
            for item in pauli:
                string[item["qubit"]] = item["pauli"][-1]
            pauli_strings.append(("".join(reversed(string)), coef))

        observable = SparsePauliOp.from_list(pauli_strings)

        isa_circuit = self.pm.run(self.circuit)
        isa_observables = observable.apply_layout(isa_circuit.layout)
        estimator = Estimator(mode=self.backend)
        job = estimator.run([(isa_circuit, isa_observables)])
        self.exp_value_result = [float(job.result()[0].data.evs)]

    def connect(self):
        self.clear()

        return super().configure(
            num_qubits=self.num_qubits,
            execution_managed_by_target={
                "sample": "Basic",
                "exp_value": "Basic",
            },
        )
