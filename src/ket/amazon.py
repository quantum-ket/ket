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
    def __init__(self, num_qubits: int, device: Optional[str] = None):
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

        self.sample_results_by_index: Dict[int, List[int]] = {}
        self.executed_operation_indices: List[int] = []

    def clear(self):
        self.circuit = Circuit()
        self.sample_results_by_index.clear()
        self.executed_operation_indices.clear()
        self.result = None

    def submit_execution(self, logical_circuit, _, parameters=None):
        self.process_instructions(logical_circuit)

    def get_result(self):
        final_samples_list_of_lists: List[List[int]] = []

        # Para manter a ordem, você pode precisar saber a ordem dos 'index'
        # conforme foram definidos no `logical_circuit` original.
        # self.executed_operation_indices pode dar a ordem de chamada via process_instructions.
        # Se a ordem não for crítica ou se os índices forem sempre sequenciais e baixos,
        # ordenar as chaves pode ser uma aproximação.
        # A forma mais robusta seria extrair os 'index' das instruções 'Sample'
        # do 'logical_circuit' (passado para submit_execution) na ordem correta.

        # Assumindo que a ordem em executed_operation_indices é a correta ou suficiente:
        for op_index in self.executed_operation_indices:
            if op_index in self.sample_results_by_index:
                final_samples_list_of_lists.append(
                    self.sample_results_by_index[op_index]
                )
            # Se houver outros tipos de resultados (exp_values, measurements) que também usam índices,
            # você precisaria de uma lógica similar para eles e listas separadas.

        results_dict = {
            "measurements": [],  # Preencha se você tiver medições explícitas
            "exp_values": [],  # Preencha se você tiver valores esperados (ex: [self.exp_result_val])
            "samples": [list(zip(*self.result.items()))],  # Lista de listas de inteiros
            "dumps": [],  # Preencha se você tiver dumps
            "gradients": None,  # Preencha se você tiver gradientes
        }

        # print(f"DEBUG: AmazonBraket.get_result is returning: {results_dict}") # Para depuração
        return results_dict

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

    @staticmethod
    def from_aws_to_ket(state, qubits, aws_map):
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
            "u4_gate_type": "CX",
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
            direct_sample_exp_value=1024,
            qpu=qpu_params,
        )
