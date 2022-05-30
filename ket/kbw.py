from __future__ import annotations
from ctypes import *
from os import environ
from os.path import dirname


def load_kbw():
    default = {
        'KBW_PATH': dirname(__file__)+"/lib/libkbw.so",
        'KBW_MODE': 'DENSE',
        'KBW_DATA': 'DIRECT',
    }

    valid = {
        'KBW_MODE': ['DENSE', 'SPARSE'],
        'KBW_DATA': ['DIRECT', 'JSON', 'BIN'],
    }

    for name in default:
        if name not in environ:
            environ[name] = default[name]

    for name in valid:
        environ[name] = environ[name].upper()
        if environ[name] not in valid[name]:
            raise RuntimeError(
                f'Invalid KBW configuration; expecting {valid[name]}, found {environ[name]}')

    return cdll.LoadLibrary(environ['KBW_PATH'])


class kbw_error(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


kbw = load_kbw()
kbw_error_message = kbw.kbw_error_message
kbw_error_message.argtypes = [POINTER(c_size_t)]
kbw_error_message.restype = POINTER(c_ubyte)


def kbw_error_warpper(error: c_int):
    if error == 1:
        size = c_size_t()
        error_msg = kbw_error_message(size)
        raise kbw_error(bytearray(error_msg[:size.value]).decode())


class Result:
    kbw_result_delete = kbw.kbw_result_delete
    kbw_result_delete.argtypes = [c_void_p]
    kbw_result_delete.restype = None

    kbw_result_get = kbw.kbw_result_get
    kbw_result_get.argtypes = [c_void_p, POINTER(c_size_t)]
    kbw_result_get.restype = POINTER(c_ubyte)

    def __init__(self, ptr, is_json):
        self._as_parameter_ = ptr

        if ptr is None:
            size = c_size_t()
            error_msg = kbw_error_message(size)
            raise kbw_error(bytearray(error_msg[:size.value]).decode())

        self.is_json = is_json

    def __del__(self):
        if self._as_parameter_ is not None:
            self.kbw_result_delete(self)

    def get(self):
        size = c_size_t()
        data = self.kbw_result_get(self, size)
        data = bytearray(data[:size.value])
        if self.is_json:
            return data.decode()
        else:
            return data


kbw_run_dense_from_process = kbw.kbw_run_dense_from_process
kbw_run_dense_from_process.argtypes = [c_void_p]

kbw_run_dense_from_json = kbw.kbw_run_dense_from_json
kbw_run_dense_from_json.argtypes = [
    POINTER(c_ubyte), c_size_t, POINTER(c_ubyte), c_size_t]
kbw_run_dense_from_json.restype = c_void_p

kbw_run_dense_from_bin = kbw.kbw_run_dense_from_bin
kbw_run_dense_from_bin.argtypes = [
    POINTER(c_ubyte), c_size_t, POINTER(c_ubyte), c_size_t]
kbw_run_dense_from_bin.restype = c_void_p

kbw_run_sparse_from_process = kbw.kbw_run_sparse_from_process
kbw_run_sparse_from_process.argtypes = [c_void_p]

kbw_run_sparse_from_json = kbw.kbw_run_sparse_from_json
kbw_run_sparse_from_json.argtypes = [
    POINTER(c_ubyte), c_size_t, POINTER(c_ubyte), c_size_t]
kbw_run_sparse_from_json.restype = c_void_p

kbw_run_sparse_from_bin = kbw.kbw_run_sparse_from_bin
kbw_run_sparse_from_bin.argtypes = [
    POINTER(c_ubyte), c_size_t, POINTER(c_ubyte), c_size_t]
kbw_run_sparse_from_bin.restype = c_void_p


def run_dense_from_process(process):
    kbw_error_warpper(kbw_run_dense_from_process(process))


def run_sparse_from_process(process):
    kbw_error_warpper(kbw_run_sparse_from_process(process))


def run(process, get_q_code, get_metrics, is_json, set_result, func):
    q_code = get_q_code(process)
    q_code_size = len(q_code)
    q_code = (c_ubyte*q_code_size).from_buffer(bytearray(q_code))
    metrics = get_metrics(process)
    metrics_size = len(metrics)
    metrics = (c_ubyte*metrics_size).from_buffer(bytearray(metrics))
    result = Result(func(q_code, q_code_size,
                    metrics, metrics_size), is_json=is_json).get()
    set_result(process, result)


def run_from_json(process, func):
    run(process,
        lambda process: process.get_quantum_code_as_json().encode(),
        lambda process: process.get_metrics_as_json().encode(),
        True,
        lambda process, result: process.set_quantum_result_from_json(result),
        func)


def run_from_bin(process, func):
    run(process,
        lambda process: process.get_quantum_code_as_bin(),
        lambda process: process.get_metrics_as_bin(),
        False,
        lambda process, result: process.set_quantum_result_from_bin(result),
        func)


def run_dense_from_json(process):
    run_from_json(process, kbw_run_dense_from_json)


def run_sparse_from_json(process):
    run_from_json(process, kbw_run_sparse_from_json)


def run_dense_from_bin(process):
    run_from_bin(process, kbw_run_dense_from_bin)


def run_sparse_from_bin(process):
    run_from_bin(process, kbw_run_sparse_from_bin)


select = {
    'DENSE': {
        'DIRECT': run_dense_from_process,
        'JSON': run_dense_from_json,
        'BIN': run_dense_from_bin,
    },
    'SPARSE': {
        'DIRECT': run_sparse_from_process,
        'JSON': run_sparse_from_json,
        'BIN': run_sparse_from_bin,
    }
}


def execute(process):
    select[environ['KBW_MODE']][environ['KBW_DATA']](process)
