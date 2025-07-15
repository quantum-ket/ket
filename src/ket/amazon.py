"""AmazonBraket Interface for Ket

The class :class:`~ket.amazon.AmazonBraket` provides a backend to connect Ket
with Amazon Braket, the fully managed quantum computing service from AWS.

This integration allows you to run quantum circuits developed in Ket on the
diverse range of simulators and quantum processing units (QPUs) offered
through the Braket service. By acting as a bridge between Ket's high-level
programming environment and Amazon's cloud resources. This makes it
an excellent choice for experiments that require high-performance simulation
or access to different QPU architectures.

:Example:

.. code-block:: python

    from ket import *
    from ket.amazon import AmazonBraket
    from math import sqrt

    device = AmazonBraket()
    # device = AmazonBraket('arn:aws:braket:::device/quantum-simulator/amazon/tn1')
    # device = AmazonBraket('arn:aws:braket:::device/quantum-simulator/amazon/dm1')
    # device = AmazonBraket('arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1')
    # device = AmazonBraket('arn:aws:braket:us-east-1::device/qpu/ionq/Aria-2')
    # device = AmazonBraket('arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1')
    # device = AmazonBraket('arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1')
    # device = AmazonBraket('arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet')
    # device = AmazonBraket('arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3')

    process = Process(device)
    a, b = process.alloc(2)

    X(a + b)
    CNOT(H(a), b)

    with ham():
        a0 = Z(a)
        a1 = X(a)
        b0 = -(X(b) + Z(b)) / sqrt(2)
        b1 = (X(b) - Z(b)) / sqrt(2)
        h = a0 * b0 + a0 * b1 + a1 * b0 - a1 * b1

    print(exp_value(h).get())


To use the Braket QPUs and on demand simulators, you need to have an
AWS account and the necessary permissions to access the Braket service.
You can find more information on how to set up your AWS account and
permissions in the `Amazon Braket documentation <https://docs.aws.amazon.com/braket/>`_.

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
    tests and large-scale quantum experiments. Only Gate Model QPUs and
    simulators are supported.

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
        device: The ARN (Amazon Resource Name) string of the
            Braket device (QPU or simulator) to be used for execution. If
            ``None``, it defaults to using the local Braket simulator.
        shots: The number of shots for the execution to estimate
            the expectation values of an Hamiltonian term. If ``classical_shadows``
            and ``shots`` are not specified, it defaults to 2048.
        classical_shadows: If specified, it will use the classical
            shadows technique for state estimation.
        kwargs: Additional keyword arguments to be passed to the Braket device.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        shots: int | None = None,
        classical_shadows: dict | None = None,
        **kwargs,
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

        self.device = (
            AwsDevice(device, **kwargs)
            if device is not None
            else LocalSimulator(**kwargs)
        )
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
            self.coupling_graph = list(
                (i - 1, j - 1) for i, j in self.device.topology_graph.edges
            )
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
        assert len(control) <= 1, "Control qubits are not supported"
        if len(control) == 0:
            self.circuit.x(target)
        else:
            self.circuit.cnot(control[0], target)

    def pauli_y(self, target, control):
        assert len(control) == 0, "Control qubits are not supported"
        self.circuit.y(target)

    def pauli_z(self, target, control):
        assert len(control) == 0, "Control qubits are not supported"
        self.circuit.z(target)

    def hadamard(self, target, control):
        assert len(control) == 0, "Control qubits are not supported"
        self.circuit.h(target)

    def rotation_x(self, target, control, **kwargs):
        assert len(control) == 0, "Control qubits are not supported"
        match kwargs:
            case {"Ref": {"index": index, "multiplier": multiplier, "value": _}}:
                value = self.parameters[index] * multiplier
            case {"Value": value}:
                ...

        self.circuit.rx(target=target, angle=value)

    def rotation_y(self, target, control, **kwargs):
        assert len(control) == 0, "Control qubits are not supported"
        match kwargs:
            case {"Ref": {"index": index, "multiplier": multiplier, "value": _}}:
                value = self.parameters[index] * multiplier
            case {"Value": value}:
                ...

        self.circuit.ry(target=target, angle=value)

    def rotation_z(self, target, control, **kwargs):
        assert len(control) == 0, "Control qubits are not supported"
        match kwargs:
            case {"Ref": {"index": index, "multiplier": multiplier, "value": _}}:
                value = self.parameters[index] * multiplier
            case {"Value": value}:
                ...

        self.circuit.rz(target=target, angle=value)

    def phase(self, target, control, **kwargs):
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
        self.clear()

        return super().configure(
            num_qubits=self.num_qubits,
            direct_sample_exp_value=self.shots,
            classical_shadows_exp_value=self.classical_shadows,
            qpu={
                "coupling_graph": self.coupling_graph,
                "u4_gate_type": "CX",
                "u2_gate_set": "All",
            },
        )
