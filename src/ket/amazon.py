"""AmazonBraket Interface for Ket

The class :class:~ket.amazon.AmazonBraket provides a backend to connect Ket
with Amazon Braket, the fully managed quantum computing service from AWS.

This integration allows you to run quantum circuits developed in Ket on the
diverse range of simulators and quantum processing units (QPUs) offered
through the Braket service. By acting as a bridge between Ket's high-level
programming environment and Amazon's cloud resources. This makes it
an excellent choice for experiments that require high-performance simulation
or access to different QPU architectures.
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2025 Ruan Luiz Molgero Lopes <ruan.molgero@grad.ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from json import loads
from typing import Optional, List, Dict
from .clib.libket import BatchExecution

try:
    # pip3 install amazon-braket-sdk
    from braket.aws import AwsDevice
    from braket.circuits import Circuit
    from braket.devices import LocalSimulator

    BRAKET_AVAILABLE = True
except ImportError:
    BRAKET_AVAILABLE = False


class AmazonBraket(BatchExecution):  # pylint: disable=too-many-instance-attributes
    """Amazon Braket Backend for Ket.

    This class provides an interface to run Ket  quantum circuits on the Amazon
    Braket service. It enables access to a wide range of cloud-based simulators
    and real quantum hardware (QPUs), making it suitable for both small-scale
    tests and large-scale quantum experiments.

    Args:
        device (Optional[str]): The ARN (Amazon Resource Name) string of the
            Braket device (QPU or simulator) to be used for execution. If
            "None", it defaults to using the local Braket simulator.
        shots (Optional[int]): The number of shots for the execution to estimate
            the expectation values of an Hamiltonian term. If not specified, it defaults
            to 2048.
        classical_shadows (Optional[dict]): If specified, it will use the classical
            shadows technique for state estimation. Dictionary should be in the
            format: {"bias": tuple[int], "samples": int, "shots": int}.

    """

    def __init__(
        self,
        device: Optional[str] = None,
        shots: int | None = None,
        classical_shadows: dict | None = None,
    ):
        if not BRAKET_AVAILABLE:
            raise RuntimeError(
                "Amazon-Braket is not available. Please install it with: pip install \
                    ket-lang[amazon]"
            )

        if shots is not None and classical_shadows is not None:
            raise ValueError(
                "You cannot specify both 'shots' and 'classical_shadows'. "
                "Please choose one of them."
            )

        super().__init__()

        self.device = AwsDevice(device) if device is not None else LocalSimulator()
        self.device_paradigm = loads(self.device.properties.json())["paradigm"]

        if not (
            self.device_paradigm["braketSchemaHeader"]["name"]
            == "braket.device_schema.gate_model_qpu_paradigm_properties"
            or self.device_paradigm["braketSchemaHeader"]["name"]
            == "braket.device_schema.simulators.gate_model_simulator_paradigm_properties"
        ):
            raise ValueError(
                "The device must be a Gate Model QPU or Simulator. "
                f"Received: {self.device_paradigm['braketSchemaHeader']['name']}"
            )

        self.num_qubits = self.device_paradigm["qubitCount"]

        if (
            hasattr(self.device, "topology_graph")
            and self.device.topology_graph is not None
        ):
            self.coupling_graph = list(device.topology_graph.edges)
        else:
            self.coupling_graph = None

        self.circuit = Circuit()
        self.result = None

        self.sample_results_by_index: Dict[int, List[int]] = {}
        self.executed_operation_indices: List[int] = []

        self.parameters = None

        self.shots = 2048 if shots is None and classical_shadows is None else shots
        self.classical_shadows = classical_shadows

    def clear(self):
        self.circuit = Circuit()
        self.sample_results_by_index.clear()
        self.executed_operation_indices.clear()
        self.result = None
        self.parameters = None

    def submit_execution(self, circuit, parameters):
        self.parameters = parameters
        self.process_instructions(circuit)

    def get_result(self):
        results_dict = {
            "measurements": [],
            "exp_values": [],
            "samples": [list(zip(*self.result.items()))],
            "dumps": [],
            "gradients": None,
        }

        return results_dict

    def pauli_x(self, target, control):
        """Apply a Pauli-X gate to the target qubit."""
        assert len(control) <= 1, "Control qubits are not supported"
        if len(control) == 0:
            self.circuit.x(target)
        else:
            self.circuit.cnot(control[0], target)

    def pauli_y(self, target, control):
        """Apply a Pauli-Y gate to the target qubit."""
        assert len(control) == 0, "Control qubits are not supported"
        self.circuit.y(target)

    def pauli_z(self, target, control):
        """Apply a Pauli-Z gate to the target qubit."""
        assert len(control) == 0, "Control qubits are not supported"
        self.circuit.z(target)

    def hadamard(self, target, control):
        """Apply a Hadamard gate to the target qubit."""
        assert len(control) == 0, "Control qubits are not supported"
        self.circuit.h(target)

    def rotation_x(self, target, control, **kwargs):
        """Apply a rotation around the X-axis to the target qubit."""
        assert len(control) == 0, "Control qubits are not supported"
        match kwargs:
            case {"Ref": {"index": index, "multiplier": multiplier, "value": _}}:
                value = self.parameters[index] * multiplier
            case {"Value": value}:
                ...

        self.circuit.rx(target=target, angle=value)

    def rotation_y(self, target, control, **kwargs):
        """Apply a rotation around the Y-axis to the target qubit."""
        assert len(control) == 0, "Control qubits are not supported"
        match kwargs:
            case {"Ref": {"index": index, "multiplier": multiplier, "value": _}}:
                value = self.parameters[index] * multiplier
            case {"Value": value}:
                ...

        self.circuit.ry(target=target, angle=value)

    def rotation_z(self, target, control, **kwargs):
        """Apply a rotation around the Z-axis to the target qubit."""
        assert len(control) == 0, "Control qubits are not supported"
        match kwargs:
            case {"Ref": {"index": index, "multiplier": multiplier, "value": _}}:
                value = self.parameters[index] * multiplier
            case {"Value": value}:
                ...

        self.circuit.rz(target=target, angle=value)

    def phase(self, target, control, **kwargs):
        """Apply a phase gate to the target qubit."""
        match kwargs:
            case {"Ref": {"index": index, "multiplier": multiplier, "value": _}}:
                value = self.parameters[index] * multiplier
            case {"Value": value}:
                ...

        self.circuit.phaseshift(target, value)

    @staticmethod
    def _from_braket_to_ket(state, qubits, aws_map):
        """Convert a Braket state bitstring to a Ket integer result.

        Args:
            state: The bitstring state from Braket measurement, e.g., "101".
            qubits: The list of qubits as seen by Ket.
            aws_map: A map from the qubit index to its position in the Braket bitstring.

        Returns:
            An integer representing the measured state in Ket's qubit order.
        """
        return int(
            "".join([state[aws_map[q]] if q in aws_map else "0" for q in qubits]),
            2,
        )

    def sample(self, _, qubits, shots: int):
        result = self.device.run(self.circuit, shots=shots).result()
        aws_map = {q: i for i, q in enumerate(result.measured_qubits)}
        self.result = {
            self._from_braket_to_ket(k, qubits, aws_map): v
            for k, v in result.measurement_counts.items()
        }

    def connect(self):
        """Configures a Process to use AmazonBraket.

        The return value of this function must be passed to a :class:`~ket.base.Process`
        constructor. The result should not be reused, and the AmazonBraket instance must
        remain alive until the end of the process's lifetime.
        """

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
            qpu=qpu_params,
        )
