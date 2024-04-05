"""Wrapper for KBW C API."""

from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from ctypes import c_void_p, c_size_t, c_bool, c_uint32
from typing import Literal
from os import environ
from os.path import dirname
from .wrapper import load_lib, os_lib_name

API_argtypes = {
    "kbw_set_log_level": ([c_uint32], []),
    "kbw_make_configuration": ([c_size_t, c_bool, c_bool, c_bool], [c_void_p]),
}


def kbw_path():
    """Get KBW path"""

    if "KBW_PATH" in environ:
        path = environ["KBW_PATH"]
    else:
        path = f'{dirname(__file__)}/libs/{os_lib_name("kbw")}'

    return path


API = load_lib("KBW", kbw_path(), API_argtypes, "kbw_error_message")


def set_log(level: int):
    """Set KBW log level"""

    API["kbw_set_log_level"](level)


def get_simulator(
    num_qubits: int,
    execution: Literal["live", "batch"] = "live",
    simulator: Literal["sparse", "dense"] = "sparse",
    decompose: bool = False,
):
    """Create a configuration"""

    return API["kbw_make_configuration"](
        num_qubits,
        execution == "live",
        simulator == "sparse",
        decompose,
    )
