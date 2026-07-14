"""AmazonBraket Interface for Ket

The class :class:`~ket.amazon.AmazonBraket` provides a backend to connect Ket
with Amazon Braket, the fully managed quantum computing service from AWS.

This integration allows you to run quantum circuits developed in Ket on the
diverse range of simulators and quantum processing units (QPUs) offered
through the Braket service. By acting as a bridge between Ket's high-level
programming environment and Amazon's cloud resources. This makes it
an excellent choice for experiments that require high-performance simulation
or access to different QPU architectures.

Example:

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

Note:
    This module requires additional dependencies from ``ket-lang[amazon]``.

    Install with: ``pip install ket-lang[amazon]``.
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2025 Ruan Luiz Molgero Lopes <ruan.molgero@grad.ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from functools import reduce
from json import loads
from operator import add
from .clib.libket.execution import BatchExecution

try:
    # pip3 install amazon-braket-sdk
    from braket.aws import AwsDevice
    from braket.circuits import Circuit, Observable
    from braket.devices import LocalSimulator
    from braket.program_sets import CircuitBinding, ProgramSet

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

    The ``shots`` argument controls how many shots are used for expectation
    value estimation.

    Args:
        device: The ARN (Amazon Resource Name) string of the
            Braket device (QPU or simulator) to be used for execution. If
            ``None``, it defaults to using the local Braket simulator.
        shots: The number of shots for the expectation values of an Hamiltonian.
        kwargs: Additional keyword arguments to be passed to the Braket device.
    """

    def __init__(
        self,
        device: str | None = None,
        shots: int = 1024,
        gradient: bool = False,
        **kwargs,
    ):
        if not BRAKET_AVAILABLE:
            raise RuntimeError(
                "Amazon-Braket is not available. Please install it with: pip install \
                    ket-lang[amazon]"
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
        self.shots = shots
        self.gradient = gradient

    @staticmethod
    def _call_gate(circuit, gate, target):  # pylint: disable=too-many-branches
        match gate:
            case "Hadamard":
                circuit.h(target)
            case "PauliX":
                circuit.x(target)
            case "PauliY":
                circuit.y(target)
            case "PauliZ":
                circuit.z(target)
            case {"RotationX": {"Value": angle}}:
                circuit.rx(target=target, angle=angle)
            case {
                "RotationX": {
                    "Ref": {
                        "multiplier": multiplier,
                        "value": value,
                        "index": _,
                    }
                }
            }:
                circuit.rx(target=target, angle=multiplier * value)
            case {"RotationY": {"Value": angle}}:
                circuit.ry(target=target, angle=angle)
            case {
                "RotationY": {
                    "Ref": {
                        "multiplier": multiplier,
                        "value": value,
                        "index": _,
                    }
                }
            }:
                circuit.ry(target=target, angle=multiplier * value)
            case {"RotationZ": {"Value": angle}}:
                circuit.rz(target=target, angle=angle)
            case {
                "RotationZ": {
                    "Ref": {
                        "multiplier": multiplier,
                        "value": value,
                        "index": _,
                    }
                }
            }:
                circuit.rz(target=target, angle=multiplier * value)
            case {"Phase": {"Value": angle}}:
                circuit.phaseshift(target=target, angle=angle)
            case {
                "Phase": {
                    "Ref": {
                        "multiplier": multiplier,
                        "value": value,
                        "index": _,
                    }
                }
            }:
                circuit.phaseshift(target=target, angle=multiplier * value)
            case _:
                raise RuntimeError(f"Undefined gate '{gate}'")

    def _build_circuit(self, gates):
        circuit = Circuit()

        for gate_inst in gates:
            gate = gate_inst["gate"]
            target = gate_inst["target"]
            decomposed = gate_inst["decomposed"]

            if decomposed is not None:
                for gate in decomposed:
                    match gate:
                        case {"CNOT": (control, target)}:
                            circuit.cnot(control, target)
                        case {"U": (gate, target)}:
                            self._call_gate(circuit, gate, target)
            else:
                self._call_gate(circuit, gate, target)

        return circuit

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

    def sample(self, gates, qubits_to_sample, shots):
        """Execute the circuit and sample the given qubits.

        .. warning::
            This method is called by Libket and should not be called directly.
        """
        circuit = self._build_circuit(gates)

        result = self.device.run(circuit, shots=shots).result()
        aws_map = {q: i for i, q in enumerate(result.measured_qubits)}

        counts = {
            self._from_braket_to_ket(k, qubits_to_sample, aws_map): v
            for k, v in result.measurement_counts.items()
        }

        mask_64bit = (1 << 64) - 1
        num_chunks = (len(qubits_to_sample) + 63) // 64

        states, shot_counts = zip(*counts.items())
        return [
            [(s >> (64 * i)) & mask_64bit for i in range(num_chunks)] for s in states
        ], list(shot_counts)

    def exp_value(self, gates, hamiltonian_list):
        """Execute the circuit and compute expectation values.

        .. warning::
            This method is called by Libket and should not be called directly.
        """
        gates_dict = {
            "X": Observable.X,
            "Y": Observable.Y,
            "Z": Observable.Z,
        }

        circuit = self._build_circuit(gates)
        observables = []

        for hamiltonian in hamiltonian_list:
            terms = []
            for pauli, coef in zip(
                hamiltonian["pauli_strings"], hamiltonian["coefficients"]
            ):

                obs_term = Observable.TensorProduct(
                    gates_dict[item["pauli"][-1]](item["qubit"]) for item in pauli
                )

                terms.append((obs_term, coef))

            observables.append(reduce(add, (c * p for p, c in terms)))

        binding = CircuitBinding(circuit, [], observables=observables)
        program_set = ProgramSet(binding)

        result = [
            float(result.expectation)
            for result in self.device.run(program_set, shots=self.shots).result()[0]
        ]

        return result

    def connect(self):
        """Connect to the device and return the configuration."""
        return self.configure(self.num_qubits, gradient=self.gradient, decompose=True)
