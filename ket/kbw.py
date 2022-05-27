from __future__ import annotations
# Licensed under the Apache License, Version 2.0;
# Copyright 2022 Evandro Chagas Ribeiro da Rosa
from ctypes import *
from os import environ
from os.path import dirname


def load_kbw():
    if "KBW_PATH" in environ:
        kbw_path = environ["KBW_PATH"]
    else:
        kbw_path = dirname(__file__)+"/lib/libkbw.so"

    return cdll.LoadLibrary(kbw_path)


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


def kbw_error_warpper_addr(addr: c_void_p):
    if addr == None:
        size = c_size_t()
        error_msg = kbw_error_message(size)
        raise kbw_error(bytearray(error_msg[:size.value]).decode())
    else:
        return addr


kbw_run_dense_from_process = kbw.kbw_run_dense_from_process
kbw_run_dense_from_process.argtypes = [c_void_p]


def execute(process):
    kbw_error_warpper(kbw_run_dense_from_process(process))
