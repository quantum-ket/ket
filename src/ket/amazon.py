from __future__ import annotations

# SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2025 Ruan Luiz Molgero Lopes <ruan.molgero@grad.ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Optional, List, Dict
from .clib.libket import BatchExecution

try:
    # pip3 install amazon-braket-sdk
    from braket.aws import AwsDevice
    from braket.circuits import Circuit  # , Instruction, gates, Observable, Gate
    from braket.devices import LocalSimulator

    BRAKET_AVAILABLE = True
except ImportError:
    BRAKET_AVAILABLE = False


class AmazonBraket(BatchExecution):
    def __init__(
        self,
        num_qubits: int,
        device: Optional[str] = None
    ):
        if not BRAKET_AVAILABLE:
            raise RuntimeError(
                "Amazon-Braket is not available. Please install it with: pip install ket-lang[amazon]"
            )

        super().__init__()

        self.num_qubits = num_qubits
        self.device = AwsDevice(device) if device else LocalSimulator()

        self.circuit = Circuit()
        self.result = None
        # self.exp_result = None

    def clear(self):
        self.circuit = Circuit()
        # self.exp_result = None

    def submit_execution(self, logical_circuit, _, parameters=None):
        self.process_instructions(logical_circuit)

    def get_result(self):
        return {
            "measurements": [],
            "exp_values": [],
            "samples": [],
            "dumps": [],
            "gradients": None,
        }

    def pauli_x(self, target, control):
        """Apply a Pauli-X gate to the target qubit."""
        assert len(control) <= 1, "Control qubits are not supported"
        if len(control) == 0:
            self.circuit.x(target)
        else:
            self.circuit.cnot(control[0], target)

    # def pauli_y(self, target, control):
    #     pass

    # def pauli_z(self, target, control):
    #     pass

    def hadamard(self, target, control):
        """Apply a Hadamard gate to the target qubit."""
        assert len(control) == 0, "Control qubits are not supported"
        self.circuit.h(target)

    def sample(self, _, qubits, shots):
        self.result = self.device.run(
            self.circuit, shots=shots).result().measurement_counts
        
        # return self.result

    def connect(self):
        """Configures a Process to use AmazonBraket.

        The return value of this function must be passed to a :class:`~ket.base.Process`
        constructor. The result should not be reused, and the AmazonBraket instance must remain alive
        until the end of the process's lifetime.
        """

        self.clear()

        "Unsupported"

        exec_params = {
            # "measure": "Unsupported",
            "sample": "Basic",
            # "exp_value": "Unsupported",
            # "dump":"Unsupported"
        }

        qpu_params = {
            "coupling_graph": None,
            "u4_gate_type":"CX",
            "u2_gate_set": "All",
        }
        # if self.device != LocalSimulator():
        #     qpu_params = {
        #         "coupling_graph": None,      # This was in your original call
        #         "u4_gate_type": "CX",        # This was in your original call, libket.py calls it u4_gate
        #         "u2_gates": "All"          # This was in your original call, libket.py calls it u2_gates
        #     }

        return super().configure(
            num_qubits=self.num_qubits,
            execution_managed_by_target=exec_params,
            qpu=qpu_params
        )