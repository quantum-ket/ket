""" Module providing functionality to interact with IBM Quantum and IBM Cloud devices.

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

try:
    from qiskit import QuantumCircuit
    from qiskit.providers import Backend
except ImportError as exc:
    raise ImportError(
        "IBMDevice requires the qiskit module to be used. You can install them"
        "alongside ket by running `pip install ket[ibm]`."
    ) from exc

import json
from ctypes import CFUNCTYPE, POINTER, c_uint8, c_size_t
from ..clib.libket import BatchCExecution, make_configuration
from .ibm_client import IBMClient

__all__ = ["IBMDevice"]


class IBMDevice:
    """IBM Qiskit backend for Ket process.

    Args:
        backend: The backend to be used for the quantum execution.
        num_qubits: The number of qubits to be used for the quantum circuit.
        use_qiskit_transpiler: Use Qiskit transpiler instead of Ket's.
    """

    def __init__(
        self,
        backend: Backend,
        num_qubits: int | None = None,
        *,
        use_qiskit_transpiler: bool = False,
    ) -> None:
        self.num_qubits = (
            num_qubits if num_qubits is not None else backend.configuration().n_qubits
        )
        self.client = IBMClient(self.num_qubits, backend)

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

        self._result_json = None
        self._result_json_len = None
        self._formatted_result = None

        @CFUNCTYPE(None, POINTER(c_uint8), c_size_t, POINTER(c_uint8), c_size_t)
        def submit_execution(
            logical_circuit,
            logical_circuit_size,
            physical_circuit,
            physical_circuit_size,
        ):
            """Sends the ket circuit instructions from libket to the IBM Client."""
            logical_circuit = json.loads(
                bytearray(logical_circuit[:logical_circuit_size])
            )
            physical_circuit = json.loads(
                bytearray(physical_circuit[:physical_circuit_size])
            )
            self._formatted_result = self.client.process_instructions(
                physical_circuit if physical_circuit is not None else logical_circuit
            )

        @CFUNCTYPE(None, POINTER(POINTER(c_uint8)), POINTER(c_size_t))
        def get_result(result_ptr, size):
            """Sends the processing result from the IBM Client back to libket."""
            result_dict = self._formatted_result
            result_json = json.dumps(result_dict).encode("utf-8")
            result_len = len(result_json)

            self._result_json = (c_uint8 * result_len)(*result_json)
            self._result_json_len = result_len
            # Set the result pointer and size, both of which must remain valid inside
            # python until the libket process has finished. That's why they're class
            # attributes.
            result_ptr[0] = self._result_json
            size[0] = self._result_json_len

        self.c_struct = BatchCExecution(
            submit_execution,
            get_result,
        )

    def configure(self):
        """Set up Ket process.
        
        Example:

            .. code-block:: python

                device = IBMDevice(backend)
                process = Process(device.configure())
            

        """

        return make_configuration(
            num_qubits=self.num_qubits,
            batch_execution=self.c_struct,
            measure="Allowed",
            sample="Allowed",
            exp_value="Allowed",
            dump="Disable",
            define_qpu=self.coupling_graph is not None,
            coupling_graph=self.coupling_graph,
            u4_gate_type="CX",
            u2_gate_set="All",
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
