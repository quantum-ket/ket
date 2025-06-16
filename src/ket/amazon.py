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


class AmazonBraket(BatchExecution):
    """Amazon Braket Backend for Ket.

    This class provides an interface to run Ket  quantum circuits on the Amazon 
    Braket service. It enables access to a wide range of cloud-based simulators 
    and real quantum hardware (QPUs), making it suitable for both small-scale 
    tests and large-scale quantum experiments.

    Args:
        num_qubits (int): The total number of qubits required for the quantum
            process.
        device (Optional[str]): The ARN (Amazon Resource Name) string of the
            Braket device (QPU or simulator) to be used for execution. If
            "None", it defaults to using the local Braket simulator.
    """

    def __init__(self, num_qubits: int, device: Optional[str] = None):
        if not BRAKET_AVAILABLE:
            raise RuntimeError(
                "Amazon-Braket is not available. Please install it with: pip install \
                    ket-lang[amazon]"
            )

        super().__init__()

        self.num_qubits = num_qubits
        self.device = AwsDevice(device) if device else LocalSimulator()

        self.circuit = Circuit()
        self.result = None
        # self.exp_result = None

        self.sample_results_by_index: Dict[int, List[int]] = {}
        self.executed_operation_indices: List[int] = []

    def clear(self):
        self.circuit = Circuit()
        self.sample_results_by_index.clear()
        self.executed_operation_indices.clear()
        self.result = None

    def submit_execution(self, circuit, _, parameters=None):
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
            case {"Value": value}:
                ...

        self.circuit.rx(target=target, angle=value)

    def rotation_y(self, target, control, **kwargs):
        """Apply a rotation around the Y-axis to the target qubit."""
        assert len(control) == 0, "Control qubits are not supported"
        match kwargs:
            case {"Value": value}:
                ...

        self.circuit.ry(target=target, angle=value)

    def rotation_z(self, target, control, **kwargs):
        """Apply a rotation around the Z-axis to the target qubit."""
        assert len(control) == 0, "Control qubits are not supported"
        match kwargs:
            case {"Value": value}:
                ...

        self.circuit.rz(target=target, angle=value)

    def phase(self, target, control, **kwargs):
        """Apply a phase gate to the target qubit."""
        match kwargs:
            case {"Value": value}:
                ...
        self.circuit.phaseshift(target, value)

    @staticmethod
    def from_aws_to_ket(state, qubits, aws_map):
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
            self.from_aws_to_ket(k, qubits, aws_map): v
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
            "coupling_graph": None,
            "u4_gate_type": "CX",
            "u2_gate_set": "All",
        }
        # TO-DO: Dynamic QPU configuration based on Braket device supportedOperations
        # if self.device != LocalSimulator():
        #     qpu_params = {
        #         "coupling_graph": None,
        #         "u4_gate_type": "CX",
        #         "u2_gates": "All"
        #     }

        return super().configure(
            num_qubits=self.num_qubits,
            direct_sample_exp_value=1024,
            qpu=qpu_params,
        )
