"""Wrapper for KBW C API."""

from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from ctypes import POINTER, c_void_p, c_size_t, c_bool, c_int32, c_uint32, c_uint8
from functools import reduce
from operator import iconcat
from typing import Literal
from os import environ
from os.path import dirname
from .wrapper import load_lib, os_lib_name

API_argtypes = {
    "kbw_set_log_level": ([c_uint32], []),
    "kbw_build_info": ([], [POINTER(c_uint8), c_size_t]),
    "kbw_make_configuration": (
        [
            c_size_t,
            c_int32,
            c_bool,
            POINTER(c_size_t),
            c_size_t,
        ],
        [c_void_p],
    ),
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


SIMULATOR = {
    "dense": 0,
    "dense v1": 0,
    "sparse": 1,
    "dense v2": 2,
}


def get_simulator(
    num_qubits: int,
    execution: Literal["live", "batch"] = "live",
    simulator: Literal["sparse", "dense", "dense v2"] = "sparse",
    coupling_graph: list[tuple[int, int]] | None = None,
):
    """Create a configuration"""

    coupling_graph_size = len(coupling_graph) if coupling_graph else 0
    if coupling_graph:
        coupling_graph = reduce(iconcat, coupling_graph, [])
        coupling_graph = (c_size_t * len(coupling_graph))(*coupling_graph)

    return API["kbw_make_configuration"](
        num_qubits,
        SIMULATOR[simulator.lower()],
        execution == "live",
        coupling_graph,
        coupling_graph_size,
    )
