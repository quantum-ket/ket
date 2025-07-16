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
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit_aer import AerSimulator

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class IBMDevice(BatchExecution):  # pylint: disable=too-many-instance-attributes
    """IBM Qiskit backend for Ket process.

    The arguments ``shots`` and ``classical_shadows`` control how the
    execution is performed for estimating expectation values of an
    Hamiltonian term. Only one of these arguments can be specified at a time.

    If ``shots`` is specified, it will run the circuit multiple times
    (the number of shots) to estimate the expectation values.
    If ``classical_shadows`` is specified, it will use the classical shadows
    technique for state estimation. The dictionary should be in the
    format: ``{"bias": (int, int, int), "samples": int, "shots": int}``.
    The ``bias`` tuple represents the bias for the randomized measurements on the
    X, Y, and Z axes, respectively. The ``samples`` is the number of
    classical shadows to be generated, and ``shots`` is the number of shots
    for each sample.

    Args:
        backend: The backend to be used for the quantum execution. If not
            provided, it defaults to the AerSimulator.
        use_qiskit_transpiler: Use Qiskit transpiler instead of Ket's.
        shots: The number of shots for the execution to estimate
            the expectation values of an Hamiltonian term. If ``classical_shadows``
            and ``shots`` are not specified, it defaults to 2048.
        classical_shadows: If specified, it will use the classical
            shadows technique for state estimation.
    """

    def __init__(
        self,
        backend: BackendV2 | None = None,
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
                "You cannot specify both 'shots' and 'classical_shadows'. Please"
                " choose one of them."
            )

        super().__init__()

        if backend is None:
            backend = AerSimulator()

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

        self.circuit = QuantumCircuit(self.num_qubits)
        self.parameters = None

        self.shots = 2048 if shots is None and classical_shadows is None else shots
        self.classical_shadows = classical_shadows

        self.qubits_from_sample = None
        self.result = None

    def clear(self):
        self.circuit = QuantumCircuit(self.num_qubits)
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
        self.circuit.measure_all()
        pm = generate_preset_pass_manager(
            backend=self.backend,
            initial_layout=(
                list(range(self.num_qubits))
                if self.coupling_graph is not None
                else None
            ),
        )
        isa_circuit = pm.run(self.circuit)
        sampler = Sampler(mode=self.backend)
        job = sampler.run([isa_circuit], shots=shots)
        self.result = job.result()[0].data.meas.get_counts()
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
