"""Wrapper for KBW C API."""

from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from ctypes import POINTER, c_uint8, c_void_p, c_size_t, c_bool, c_char_p
import json
from typing import Literal
from os import environ
from os.path import dirname
from .wrapper import load_lib, os_lib_name
from .libket import API as libket

api_argtypes = {
    "kbw_build_info": ([], [POINTER(c_uint8), c_size_t]),
    "kbw_make_configuration": (
        [
            c_size_t,  # num_qubits
            c_char_p,  # simulator
            c_bool,  # use_live
            c_bool,  # decompose
            c_bool,  # gradient
            c_char_p,  # coupling_graph_json
        ],
        [c_void_p],  # qpu_config
    ),
}


def kbw_path():
    """Get KBW path"""
    return environ.get("KBW_PATH", f'{dirname(__file__)}/libs/{os_lib_name("kbw")}')


API = load_lib(
    "KBW",
    kbw_path(),
    api_argtypes,
    libket["ket_error_message"],
    libket["ket_string_delete"],
)


def get_simulator(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    num_qubits: int,
    execution: Literal["live", "batch"] = "live",
    simulator: Literal[
        "sparse",
        "sparse v2",
        "dense",
        "dense v1",
        "dense v2",
        "dense gpu",
    ] = "dense",
    coupling_graph: list[tuple[int, int]] | None = None,
    gradient: bool = False,
    decompose: bool = False,
):
    """Create a KBW simulator configuration.

    Args:
        num_qubits: Number of qubits to simulate.
        execution: Execution mode — ``"live"`` for gate-by-gate simulation or
            ``"batch"`` for deferred execution (required for gradient support).
            Defaults to ``"live"``.
        simulator: Simulator backend to use. Available options:

            - ``"sparse"``: Sparse state-vector simulator.
            - ``"sparse v2"``: Updated sparse simulator variant.
            - ``"dense"``: Dense state-vector simulator with good multithreaded
              performance. This is the default.
            - ``"dense v1"``: Previous dense simulator variant.
            - ``"dense v2"``: Dense simulator with a smaller memory footprint.
            - ``"dense gpu"``: Dense simulator targeting most GPUs (Intel, AMD,
              Apple, and NVIDIA). Recommended for large qubit counts.

        coupling_graph: Optional list of ``(control, target)`` qubit pairs
            describing the hardware connectivity. When provided, the compiler
            restricts two-qubit gates to connected pairs.
        gradient: If ``True``, forces batch execution mode and enables
            parameter-shift gradient computation. Defaults to ``False``.
        decompose: If ``True``, Libket decomposes gates into U/CNOT primitives
            before passing them to the simulator. Defaults to ``False``.

    Returns:
        An opaque configuration pointer to be passed to the Ket
        :class:`~ket.Process`.
    """
    if gradient:
        execution = "batch"

    coupling_graph_json = json.dumps(coupling_graph).encode("utf-8")

    return API["kbw_make_configuration"](
        num_qubits,  # num_qubits
        simulator.encode("utf-8"),  # simulator
        execution != "batch",  # use_live
        decompose,  # decompose
        gradient,  # gradient
        coupling_graph_json,  # coupling_graph
    )
