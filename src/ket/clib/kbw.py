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

api_argtypes = {
    "kbw_set_log_level": ([c_uint32], []),
    "kbw_build_info": ([], [POINTER(c_uint8), c_size_t]),
    "kbw_make_configuration": (
        [
            c_size_t,  # num_qubits
            c_int32,  # simulator
            c_bool,  # use_live
            POINTER(c_size_t),  # coupling_graph
            c_size_t,  # coupling_graph_size
            c_bool,  # gradient
            c_size_t,  # sample_base
            POINTER(c_size_t),  # classical_shadows
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


API = load_lib("KBW", kbw_path(), api_argtypes, "kbw_error_message")


def set_log(level: int):
    """Set KBW log level"""

    API["kbw_set_log_level"](level)


_SIMULATOR = {
    "dense": 0,
    "dense v1": 0,
    "sparse": 1,
    "dense v2": 2,
}


_CLASSICAL_SHADOWS = {"bias": (1, 1, 1), "samples": 1_000, "shots": 2048}


def get_simulator(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    num_qubits: int,
    execution: Literal["live", "batch"] = "live",
    simulator: Literal["sparse", "dense", "dense v2"] = "sparse",
    coupling_graph: list[tuple[int, int]] | None = None,
    gradient: bool = False,
    classical_shadows=None,
    direct_sample=0,
):
    """Create a configuration"""
    if gradient:
        execution = "batch"

    coupling_graph_size = len(coupling_graph) if coupling_graph else 0

    if coupling_graph:
        qubit_in_graph = [q for edge in coupling_graph for q in edge]
        if any(q not in qubit_in_graph for q in range(num_qubits)) or any(
            q not in list(range(num_qubits)) for q in qubit_in_graph
        ):
            raise ValueError("Unreachable qubit in the coupling graph.")
        coupling_graph = reduce(iconcat, coupling_graph, [])
        coupling_graph = (c_size_t * len(coupling_graph))(*coupling_graph)

    if classical_shadows is not None:
        cs = {**_CLASSICAL_SHADOWS, **classical_shadows}
        classical_shadows = (c_size_t * 5)(
            cs["bias"][0], cs["bias"][1], cs["bias"][2], cs["samples"], cs["shots"]
        )

    return API["kbw_make_configuration"](
        num_qubits,  # num_qubits
        _SIMULATOR[simulator.lower()],  # simulator
        execution == "live",  # use_live
        coupling_graph,  # coupling_graph
        coupling_graph_size,  # coupling_graph_size
        gradient,  # gradient
        direct_sample,  # sample_base
        classical_shadows,  # classical_shadows
    )
