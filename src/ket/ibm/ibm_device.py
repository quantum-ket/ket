""" Interface for setting up connection to the IBM quantum infrastructure. """

from __future__ import annotations

# SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2024 Otávio Augusto de Santana Jatobá
# <otavio.jatoba@grad.ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

try:
    from qiskit import QuantumCircuit
    from qiskit.providers import Backend
    from qiskit_ibm_runtime import QiskitRuntimeService
except ImportError as exc:
    raise ImportError(
        "IBMDevice requires the qiskit module to be used. You can install them"
        "alongside ket by running `pip install ket[ibm]`."
    ) from exc

import json
from ctypes import CFUNCTYPE, POINTER, c_uint8, c_size_t
from ..clib.libket import BatchCExecution, make_configuration
from .ibm_client import IBMClient


class IBMDevice:
    """Sets up a ket process to connect to the IBM Quantum or IBM Cloud devices.

    Attributes:
        backend: The backend to be used for the quantum circuit.
        service: The Qiskit runtime service to be used for the quantum circuit.
        num_qubits: The number of qubits to be used for the quantum circuit.
        client: The IBMClient object to process the instructions.
    """

    def __init__(
        self,
        backend: Backend,
        service: QiskitRuntimeService | None = None,
        num_qubits: int | None = None,
        *,
        use_qiskit_mapping: bool = True,
    ) -> None:
        """
        Initializes the IBMDevice object.

        Parameters:
        ---
        backend (Backend): The backend to be used for the quantum circuit.
        service (QiskitRuntimeService, optional): The Qiskit runtime service to be used
        for the quantum circuit. Defaults to None.
        num_qubits (int, optional): The number of qubits to be used for the quantum
        circuit. Defaults to the max number of qubits available to the backend.
        """

        self.num_qubits = (
            backend.configuration().n_qubits if not num_qubits else num_qubits
        )
        self.client = IBMClient(self.num_qubits, backend, service)

        if backend.coupling_map and not use_qiskit_mapping:
            self.coupling_graph = list(backend.coupling_map.graph.edge_list())
        else:
            self.coupling_graph = None

        self.result_json = None
        self.result_json_len = None
        self._formatted_result = None

        @CFUNCTYPE(None, POINTER(c_uint8), c_size_t)
        def submit_execution(data, size):
            """Sends the ket circuit instructions from libket to the IBM Client."""
            instructions = json.loads(bytearray(data[:size]))
            self._formatted_result = self.client.process_instructions(instructions)

        @CFUNCTYPE(None, POINTER(POINTER(c_uint8)), POINTER(c_size_t))
        def get_result(result_ptr, size):
            """Sends the processing result from the IBM Client back to libket."""
            result_dict = self._formatted_result
            result_json = json.dumps(result_dict).encode("utf-8")
            result_len = len(result_json)

            self.result_json = (c_uint8 * result_len)(*result_json)
            self.result_json_len = result_len
            # Set the result pointer and size, both of which must remain valid inside
            # python until the libket process has finished. That's why they're class
            # attributes.
            result_ptr[0] = self.result_json
            size[0] = self.result_json_len

        @CFUNCTYPE(c_uint8)
        def get_status():
            return 0

        self.c_struct = BatchCExecution(
            submit_execution,
            get_result,
            get_status,
        )

    def build(self):
        """Set up the configuration for the ket process inside libket."""

        return make_configuration(
            num_qubits=self.num_qubits,
            batch_execution=self.c_struct,
            measure="Allowed",
            sample="Allowed",
            exp_value="Allowed",
            dump="Disable",
            coupling_graph=self.coupling_graph,
            u4_gate_type="CZ",
            u2_gate_set="RzSx",
        )

    @property
    def circuit(self) -> QuantumCircuit:
        """Quantum circuit object for the IBM device."""
        return self.client.circuit

    @property
    def isa_circuit(self) -> QuantumCircuit:
        """Quantum circuit object for the IBM device."""
        return self.client.isa_circuit

    @property
    def backend(self) -> Backend:
        """Backend object for the IBM device."""
        return self.client.backend

    @property
    def service(self) -> QiskitRuntimeService | None:
        """IBM runtime service used for the cloud connection."""
        return self.client.service
