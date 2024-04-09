"""IBM Quantum Devices backend for Ket"""

import json
from ctypes import CFUNCTYPE, POINTER, c_uint8, c_size_t
from ..clib.libket import BatchCExecution, API as libket
from .qiskit_client import QiskitClient

from qiskit.providers import Backend
from qiskit_ibm_runtime import QiskitRuntimeService


class QiskitInterface:
    def __init__(
        self, num_qubits: int, backend: Backend, service: QiskitRuntimeService | None
    ) -> None:
        self.num_qubits = num_qubits

        self.result_json = None
        self.result_json_len = None

        self.client = QiskitClient(num_qubits, backend, service)
        self._formatted_result = None

        @CFUNCTYPE(None, POINTER(c_uint8), c_size_t)
        def submit_execution(data, size):
            instructions = json.loads(bytearray(data[:size]))
            self._formatted_result = self.client.process_instructions(instructions)

        @CFUNCTYPE(None, POINTER(POINTER(c_uint8)), POINTER(c_size_t))
        def get_result(result_ptr, size):
            result_dict = self._formatted_result
            result_json = json.dumps(result_dict).encode("utf-8")
            result_len = len(result_json)

            self.result_json = (c_uint8 * result_len)(*result_json)
            self.result_json_len = result_len
            result_ptr[0] = self.result_json
            size[0] = self.result_json_len
            return

        @CFUNCTYPE(c_uint8)
        def get_status():
            return 0

        self.c_struct = BatchCExecution(
            submit_execution,
            get_result,
            get_status,
        )

    def make_configuration(self):
        """Make configuration"""

        return libket["ket_batch_make_configuration"](
            self.num_qubits,
            self.c_struct,
        )
