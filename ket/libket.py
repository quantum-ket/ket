from __future__ import annotations
# Licensed under the Apache License, Version 2.0;
# Copyright 2022 Evandro Chagas Ribeiro da Rosa
from ctypes import *
import weakref

EQ = 0
NEQ = 1
GT = 2
GEQ = 3
LT = 4
LEQ = 5
ADD = 6
SUB = 7
MUL = 8
MOD = 9
DIV = 10
SLL = 11
SRL = 12
AND = 13
OR = 14
XOR = 15

PAULI_X = 0
PAULI_Y = 1
PAULI_Z = 2
HADAMARD = 3
PHASE = 4
RX = 5
RY = 6
RZ = 7

SUCCESS = 0
CONTROL_TWICE = 1
DATA_NOT_AVAILABLE = 2
DEALLOCATED_QUBIT = 3
FAIL_TO_PARSE_RESULT = 4
NO_ADJ = 5
NO_CTRL = 6
NON_GATE_INSTRUCTION = 7
NOT_BIN = 8
NOT_JSON = 9
NOT_UNITARY = 10
PLUGIN_ON_CTRL = 11
TARGET_ON_CONTROL = 12
TERMINATED_BLOCK = 13
UNDEFINED_CLASSICAL_OP = 14
UNDEFINED_DATA_TYPE = 15
UNDEFINED_GATE = 16
UNEXPECTED_RESULT_DATA = 17
UNMATCHED_PID = 18
UNDEFINED_ERROR = 19

JSON = 0
BIN = 1

API_argtypes = {
    # 'ket_type_method': ([input_list], [output_list]),
    'ket_process_new': ([c_size_t], [c_void_p]),
    'ket_process_delete': ([c_void_p], []),
    'ket_process_allocate_qubit': ([c_void_p, c_bool], [c_void_p]),
    'ket_process_free_qubit': ([c_void_p, c_void_p, c_bool], []),
    'ket_process_apply_gate': ([c_void_p, c_int32, c_double, c_void_p], []),
    'ket_process_apply_plugin': ([c_void_p, c_char_p, c_char_p, POINTER(c_void_p), c_size_t], []),
    'ket_process_measure': ([c_void_p, POINTER(c_void_p), c_size_t], [c_void_p]),
    'ket_process_ctrl_push': ([c_void_p, POINTER(c_void_p), c_size_t], []),
    'ket_process_ctrl_pop': ([c_void_p], []),
    'ket_process_adj_begin': ([c_void_p], []),
    'ket_process_adj_end': ([c_void_p], []),
    'ket_process_get_label': ([c_void_p], [c_void_p]),
    'ket_process_open_block': ([c_void_p, c_void_p], []),
    'ket_process_jump': ([c_void_p, c_void_p], []),
    'ket_process_branch': ([c_void_p, c_void_p, c_void_p, c_void_p], []),
    'ket_process_dump': ([c_void_p, POINTER(c_void_p), c_size_t], [c_void_p]),
    'ket_process_add_int_op': ([c_void_p, c_int32, c_void_p, c_void_p], [c_void_p]),
    'ket_process_int_new': ([c_void_p, c_int64], [c_void_p]),
    'ket_process_int_set': ([c_void_p, c_void_p, c_void_p], []),
    'ket_process_prepare_for_execution': ([c_void_p], []),
    'ket_process_exec_time': ([c_void_p], [c_double]),
    'ket_process_set_timeout': ([c_void_p, c_uint64], []),
    'ket_process_serialize_metrics': ([c_void_p, c_int32], []),
    'ket_process_serialize_quantum_code': ([c_void_p, c_int32], []),
    'ket_process_get_serialized_metrics': ([c_void_p], [POINTER(c_uint8), c_size_t, c_int32]),
    'ket_process_get_serialized_quantum_code': ([c_void_p], [POINTER(c_uint8), c_size_t, c_int32]),
    'ket_process_set_serialized_result': ([c_void_p, POINTER(c_ubyte), c_size_t, c_int32], []),
    'ket_qubit_delete': ([c_void_p], []),
    'ket_qubit_index': ([c_void_p], [c_size_t]),
    'ket_qubit_pid': ([c_void_p], [c_size_t]),
    'ket_qubit_allocated': ([c_void_p], [c_bool]),
    'ket_qubit_measured': ([c_void_p], [c_bool]),
    'ket_dump_delete': ([c_void_p], []),
    'ket_dump_states_size': ([c_void_p], [c_size_t]),
    'ket_dump_state': ([c_void_p, c_size_t], [POINTER(c_uint64), c_size_t]),
    'ket_dump_amplitudes_real': ([c_void_p], [POINTER(c_double), c_size_t]),
    'ket_dump_amplitudes_imag': ([c_void_p], [POINTER(c_double), c_size_t]),
    'ket_dump_available': ([c_void_p], [c_bool]),
    'ket_future_delete': ([c_void_p], []),
    'ket_future_value': ([c_void_p], [c_int64]),
    'ket_future_index': ([c_void_p], [c_size_t]),
    'ket_future_pid': ([c_void_p], [c_size_t]),
    'ket_future_available': ([c_void_p], [c_bool]),
    'ket_label_delete': ([c_void_p], []),
    'ket_label_index': ([c_void_p], [c_size_t]),
    'ket_label_pid': ([c_void_p], [c_size_t]),
}

API = {}


class libket_error(Exception):
    def __init__(self, message, error_code):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


def from_u8_to_str(data, size):
    return bytearray(data[:size.value]).decode()


class APIWrapper:
    def __init__(self, call, output, ket_error_message):
        self.call = call
        self.output = output
        self.ket_error_message = ket_error_message

    def __call__(self, *args):
        out = [c_type() for c_type in self.output]
        error_code = self.call(*args, *out)
        if error_code != SUCCESS:
            size = c_size_t()
            error_msg = self.ket_error_message(error_code, size)
            raise libket_error(from_u8_to_str(error_msg, size), error_code)
        elif len(out) == 1:
            return out[0]
        elif len(out) != 0:
            return out


def load_libket():
    from os import environ
    from os.path import dirname
    global API

    if "LIBKET_PATH" in environ:
        libket_path = environ["LIBKET_PATH"]
    else:
        libket_path = dirname(__file__)+"/lib/libket.so"

    lib = cdll.LoadLibrary(libket_path)
    ket_error_message = lib.ket_error_message
    ket_error_message.argtypes = [c_int32, POINTER(c_size_t)]
    ket_error_message.restype = POINTER(c_uint8)

    API = {}
    for name in API_argtypes:
        call = lib.__getattr__(name)
        call.argtypes = [
            *API_argtypes[name][0],
            *[POINTER(t) for t in API_argtypes[name][1]]
        ]

        API[name] = APIWrapper(
            call, API_argtypes[name][1],
            ket_error_message
        )


class process:
    def __init__(self, pid: int):
        self.pid = pid
        self._as_parameter_ = API['ket_process_new'](pid)
        self._finalizer = weakref.finalize(
            self, API['ket_process_delete'], self._as_parameter_)

    def __getattr__(self, name: str):
        return lambda *args: API['ket_process_'+name](self, *args)

    def __repr__(self) -> str:
        return f"<Libket 'process' ({self.pid})>"


class qubit:
    def __init__(self, addr: c_void_p):
        self._as_parameter_ = addr
        self._finalizer = weakref.finalize(
            self, API['ket_qubit_delete'], self._as_parameter_)

    def __getattr__(self, name: str):
        return lambda *args: API['ket_qubit_'+name](self, *args)

    def __repr__(self) -> str:
        return f"<Libket 'qubit' {self.pid().value, self.index().value}>"


class libket_dump:
    def __init__(self, addr: c_void_p):
        self._as_parameter_ = addr
        self._finalizer = weakref.finalize(
            self, API['ket_dump_delete'], self._as_parameter_)

    def __getattr__(self, name: str):
        return lambda *args: API['ket_dump_'+name](self, *args)

    def __repr__(self) -> str:
        return f"<Libket 'dump' {self.pid().value, self.index().value}>"


class libket_future:
    def __init__(self, addr: c_void_p):
        self._as_parameter_ = addr
        self._finalizer = weakref.finalize(
            self, API['ket_future_delete'], self._as_parameter_)

    def __getattr__(self, name: str):
        return lambda *args: API['ket_future_'+name](self, *args)

    def __repr__(self) -> str:
        return f"<Libket 'future' {self.pid().value, self.index().value}>"


class libket_label:
    def __init__(self, addr: c_void_p):
        self._as_parameter_ = addr
        self._finalizer = weakref.finalize(
            self, API['ket_label_delete'], self._as_parameter_)

    def __getattr__(self, name: str):
        return lambda *args: API['ket_label_'+name](self, *args)

    def __repr__(self) -> str:
        return f"<Libket 'label' {self.pid().value, self.index().value}>"


def from_list_to_c_vector(data):
    return (c_void_p*len(data))(*(d._as_parameter_ for d in data)), len(data)


load_libket()

if __name__ == '__main__':
    from pprint import pprint
    import json

    PAULI_X = 0
    HADAMARD = 3

    p = process(1)

    q0 = qubit(p.allocate_qubit(False))
    q1 = qubit(p.allocate_qubit(False))

    p.apply_gate(HADAMARD, 0.0, q0)
    p.ctrl_push(*from_list_to_c_vector([q0]))
    p.apply_gate(PAULI_X, 0.0, q1)
    p.ctrl_pop()

    p.prepare_for_execution()
    p.serialize_metrics(JSON)
    p.serialize_quantum_code(JSON)

    pprint(json.loads(from_u8_to_str(*p.get_serialized_metrics()[:-1])))
    pprint(json.loads(from_u8_to_str(*p.get_serialized_quantum_code()[:-1])))
