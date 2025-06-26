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

try:
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.providers import Backend
    from qiskit.circuit import library
except ImportError as exc:
    raise ImportError(
        "IBMDevice requires the qiskit module to be used. You can install them"
        "alongside ket by running `pip install ket[ibm]`."
    ) from exc

from typing import Literal
import json
from ctypes import CFUNCTYPE, POINTER, c_uint8, c_size_t, c_double
from ..clib.libket import BatchCExecution, make_configuration, BatchExecution
from .ibm_client import IBMClient

__all__ = ["IBMDevice"]


class IBMDevice:  # pylint: disable=too-many-instance-attributes
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
        self.client = None
        self._backend = backend

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

        @CFUNCTYPE(
            None,
            POINTER(c_uint8),
            c_size_t,
            POINTER(c_uint8),
            c_size_t,
            POINTER(c_double),
            c_size_t,
        )
        def submit_execution(
            circuit,
            circuit_size,
            _parameters,
            _parameters_size,
        ):
            """Sends the ket circuit instructions from libket to the IBM Client."""
            circuit = json.loads(bytearray(circuit[:circuit_size]))
            self.client = IBMClient(self.num_qubits, self._backend)
            self._formatted_result = self.client.process_instructions(circuit)

        @CFUNCTYPE(
            None,
            POINTER(POINTER(c_uint8)),
            POINTER(c_size_t),
        )
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

        @CFUNCTYPE(None)
        def clear():
            pass

        self.c_struct = BatchCExecution(
            submit_execution,
            get_result,
            clear,
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
            execution=self,
            execution_managed_by_target={
                "measure": "Basic",
                "sample": "Basic",
                "exp_value": "Basic",
            },
            qpu=(
                None
                if self.coupling_graph is None
                else {
                    "coupling_graph": self.coupling_graph,
                    "u2_gates": "ZYZ",
                    "u4_gate": "CX",
                }
            ),
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
        return self._backend
