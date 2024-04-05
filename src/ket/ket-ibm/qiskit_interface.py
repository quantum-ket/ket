"""IBM Quantum Devices backend for Ket"""

import json
from os import path
from ctypes import CFUNCTYPE, Structure, POINTER, c_uint8, c_size_t, c_void_p
from ket.clib.wrapper import load_lib
from qiskit_client import QiskitClient
from qiskit.providers import Backend


class BatchCExecution(Structure):  # pylint: disable=too-few-public-methods
    """C BatchCExecution Structure"""

    _fields_ = [
        ("submit_execution", CFUNCTYPE(None, POINTER(c_uint8), c_size_t)),
        ("get_result", CFUNCTYPE(None, POINTER(POINTER(c_uint8)), POINTER(c_size_t))),
        ("get_status", CFUNCTYPE(c_uint8)),
    ]


class QiskitInterface:
    def __init__(self, num_qubits: int, backend: Backend) -> None:
        self.num_qubits = num_qubits

        self.result_json = None
        self.result_json_len = None

        self.client = QiskitClient(backend, num_qubits)
        self._formatted_result   = None

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

        self.lib = load_lib(
            "Ket C Execution",
            path.dirname(__file__) + "/libket_c_execution.so",
            {
                "ket_c_exec_make_configuration": (
                    [c_size_t, POINTER(BatchCExecution)],
                    [c_void_p],
                )
            },
            "ket_c_exec_error_message",
        )

    def make_configuration(self):
        """Make configuration"""

        return self.lib["ket_c_exec_make_configuration"](
            self.num_qubits,
            self.c_struct,
        )
