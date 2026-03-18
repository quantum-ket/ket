"""Wrapper for KBW C API."""

from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from ctypes import POINTER, c_void_p, c_size_t, c_bool, c_int32, c_uint32, c_uint8
from functools import reduce
from operator import iconcat
import os
from typing import Literal
from os import environ
from os.path import dirname
import warnings
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
    "sparse": 1110,
    "sparse v1": 1110,
    "sparse v2": 1210,
    "dense": 2110,
    "dense v1": 2110,
    "dense v2": 2210,
    "dense v2 block": 2211,
    "dense gpu": 2120,
    "dense gpu block": 2121,
}


def _get_simulator_code(num_qubits: int, simulator: str):
    simulator = simulator.lower()
    if simulator not in _SIMULATOR:
        warnings.warn(
            f"Unknown simulator {simulator}. "
            f'Available options {list(_SIMULATOR.keys())}. Default to "sparse"'
        )
        simulator = "sparse"

    if simulator == "dense gpu":
        try:
            kbw_block_size = int(os.environ.get("KBW_BLOCK_SIZE", "20"))
        except ValueError:
            kbw_block_size = 20
        if num_qubits > kbw_block_size:
            simulator = "dense gpu block"

    return _SIMULATOR[simulator]


_CLASSICAL_SHADOWS = {"bias": (1, 1, 1), "samples": 1_000, "shots": 2048}


def get_simulator(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    num_qubits: int,
    execution: Literal["live", "batch"] = "live",
    simulator: Literal[
        "sparse",
        "dense",
        "dense v2",
        "dense gpu",
    ] = "sparse",
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
        _get_simulator_code(num_qubits, simulator),  # simulator
        execution == "live",  # use_live
        coupling_graph,  # coupling_graph
        coupling_graph_size,  # coupling_graph_size
        gradient,  # gradient
        direct_sample,  # sample_base
        classical_shadows,  # classical_shadows
    )
