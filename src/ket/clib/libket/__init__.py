"""Wrapper for Libket C API."""

from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from ctypes import (
    CFUNCTYPE,
    Structure,
    c_uint8,
    c_void_p,
    c_size_t,
    POINTER,
    c_bool,
    c_int32,
    c_uint64,
    c_double,
    c_char_p,
)
import json
from typing import Sequence
import weakref
from os import environ
from os.path import dirname

from ..wrapper import load_lib, os_lib_name


class BatchCExecution(Structure):  # pylint: disable=too-few-public-methods
    """C BatchCExecution Structure"""

    _fields_ = [
        (
            "sample",
            CFUNCTYPE(
                c_int32,
                c_char_p,  # gates_json
                POINTER(c_size_t),  # qubits_to_sample
                c_size_t,  # qubits_to_sample_len
                c_size_t,  # shots
                POINTER(c_char_p),  # sample_json
            ),
        ),
        (
            "exp_value",
            CFUNCTYPE(
                c_int32,
                c_char_p,  # gates_json
                c_char_p,  # hamiltonian_list_json
                POINTER(c_double),  # result
            ),
        ),
        ("exp_value_available", CFUNCTYPE(c_bool)),
    ]


class LiveCExecution(Structure):  # pylint: disable=too-few-public-methods
    """C LiveCExecution Structure"""

    _fields_ = [
        (
            "compute_gate",
            CFUNCTYPE(
                c_int32,
                c_char_p,  # json
            ),
        ),
        (
            "measure",
            CFUNCTYPE(
                c_int32,
                POINTER(c_size_t),  # qubits
                c_size_t,  # len
                POINTER(c_uint64),  # result
            ),
        ),
        (
            "dump",
            CFUNCTYPE(
                c_int32,
                POINTER(c_size_t),  # qubits
                c_size_t,  # len
                POINTER(c_char_p),  # result_json
            ),
        ),
        (
            "sample",
            CFUNCTYPE(
                c_int32,
                POINTER(c_size_t),  # qubits
                c_size_t,  # len
                c_size_t,  # shots
                POINTER(c_char_p),  # result_json
            ),
        ),
        (
            "exp_value",
            CFUNCTYPE(
                c_int32,
                c_char_p,  # hamiltonian_json
                POINTER(c_double),  # result
            ),
        ),
    ]


class CNativeGateSet(Structure):
    """C CNativeGateSet Structure"""

    _fields_ = [
        (
            "translate",
            CFUNCTYPE(
                c_int32,
                c_char_p,  # gate_json
                c_size_t,  # target
                POINTER(c_char_p),  # native_gate_json
            ),
        ),
        (
            "cnot",
            CFUNCTYPE(
                c_int32,
                c_size_t,  # control
                c_size_t,  # target
                POINTER(c_char_p),  # native_gate_json
            ),
        ),
        (
            "swap",
            CFUNCTYPE(
                c_int32,
                c_size_t,  # a
                c_size_t,  # b
                POINTER(c_char_p),  # native_gate_json
            ),
        ),
    ]


api_argtypes = {
    # 'ket_type_method': ([input_list], [output_list]),
    "ket_build_info": ([], [POINTER(c_uint8), c_size_t]),
    "ket_string_delete": (
        [c_char_p],  # ptr
        [],
    ),
    "ket_block_new": (
        [],
        [c_void_p],  # block
    ),
    "ket_block_delete": (
        [c_void_p],  # block
        [],
    ),
    "ket_block_append_gate": (
        [c_void_p, c_char_p, c_size_t],  # block, gate_json, target
        [],
    ),
    "ket_block_inverse": (
        [c_void_p],  # block
        [c_void_p],  # inverted_block
    ),
    "ket_block_control": (
        [
            c_void_p,
            POINTER(c_size_t),
            c_size_t,
        ],  # block, control_qubits, control_qubits_len
        [c_void_p],  # controlled_block
    ),
    "ket_block_enable_approximated_decomposition": (
        [c_void_p],  # block
        [],
    ),
    "ket_block_lock_control": (
        [c_void_p],  # block
        [],
    ),
    "ket_block_append_block": (
        [c_void_p, c_void_p],  # block, other
        [],
    ),
    "ket_block_add_global_phase": (
        [c_void_p, c_double],  # block, phase
        [],
    ),
    "ket_block_proprieties_json": (
        [c_void_p],  # block
        [c_char_p],  # proprieties
    ),
    "ket_quantum_execution_live": (
        [POINTER(LiveCExecution), c_bool],  # live, decompose
        [c_void_p],  # quantum_execution
    ),
    "ket_quantum_execution_batch": (
        [
            POINTER(BatchCExecution),
            POINTER(CNativeGateSet),
            c_bool,
            c_char_p,
            c_char_p,
        ],  # batch, native_gate_set, gradient, coupling_graph_json
        [c_void_p],  # quantum_execution
    ),
    "ket_process_new": (
        [c_void_p],  # qpu_config
        [c_void_p],  # process
    ),
    "ket_process_delete": (
        [c_void_p],  # process
        [],
    ),
    "ket_process_alloc": (
        [c_void_p],  # process
        [c_size_t],  # qubit
    ),
    "ket_process_append_block": (
        [c_void_p, c_void_p],  # process, block
        [],
    ),
    "ket_process_measure": (
        [c_void_p, POINTER(c_size_t), c_size_t],  # process, qubits, qubits_len
        [c_uint64],  # result
    ),
    "ket_process_dump": (
        [c_void_p, POINTER(c_size_t), c_size_t],  # process, qubits, qubits_len
        [c_char_p],  # dump_json
    ),
    "ket_process_sample": (
        [
            c_void_p,
            POINTER(c_size_t),
            c_size_t,
            c_size_t,
        ],  # process, qubits, qubits_len, shots
        [c_char_p],  # sample_json
    ),
    "ket_process_read_exp_value": (
        [c_void_p],  # process
        [c_char_p],  # result_json
    ),
    "ket_process_read_gradient": (
        [c_void_p],  # process
        [c_char_p],  # result_json
    ),
    "ket_process_param": (
        [c_void_p, c_double],  # process, param
        [c_size_t],  # param_index
    ),
    "ket_process_execute": (
        [c_void_p],  # process
        [],
    ),
    "ket_process_read_sample": (
        [c_void_p],  # process
        [c_char_p],  # sample_json
    ),
    "ket_process_exp_value": (
        [c_void_p, c_char_p],  # process, hamiltonian_json
        [c_double, c_bool],  # result, some_result
    ),
    "ket_config_new": (
        [c_size_t],  # num_qubits
        [c_void_p],  # qpu_config
    ),
    "ket_process_gates_json": (
        [c_void_p],  # process
        [c_char_p],  # block
    ),
    "ket_error_message": ([c_int32], [c_char_p]),
    "ket_process_status": ([c_void_p], [c_char_p]),
    "ket_block_set_as_diagonal": ([c_void_p], []),
    "ket_block_set_as_permutation": ([c_void_p], []),
}


def libket_path():
    """Get Libket shared library path"""
    return environ.get(
        "LIBKET_PATH", f'{dirname(__file__)}/../libs/{os_lib_name("ket")}'
    )


API = load_lib("Libket", libket_path(), api_argtypes)


class HasProcess:  # pylint: disable=too-few-public-methods
    """Object with an associated quantum process."""

    def __init__(self, ket_process):
        self._ket_process = ket_process

    @property
    def ket_process(self):
        """Get Ket process."""
        return self._ket_process


def search_process(ket_process, args, kwargs):
    def inner(ket_process, arg):
        if hasattr(arg, "ket_process"):
            arg_process = arg.ket_process
            if ket_process is not None and ket_process is not arg_process:
                raise ValueError("parameter with different Ket processes")
            ket_process = arg_process
        return ket_process

    def search(ket_process, args):
        for arg in args:
            if isinstance(arg, Sequence) and not isinstance(arg, str):
                for subarg in arg:
                    ket_process = inner(ket_process, subarg)
            else:
                ket_process = inner(ket_process, arg)
        return ket_process

    if ket_process is None:
        ket_process = search(ket_process, args)
        ket_process = search(ket_process, kwargs.values())

    if ket_process is None:
        raise ValueError("Ket process not found in the parameters")

    return ket_process


class Process(HasProcess):
    """Libket process wrapper from C API"""

    def __init__(self, configuration):
        super().__init__(ket_process=self)

        self._as_parameter_ = API["ket_process_new"](configuration)
        self._finalizer = weakref.finalize(
            self, API["ket_process_delete"], self._as_parameter_
        )

    def __getattr__(self, name: str):
        return lambda *args: API["ket_process_" + name](self, *args)

    def __repr__(self) -> str:
        return f"<Libket 'process', pid={hex(id(self.ket_process))}>"


class Block(HasProcess):
    def __init__(self, ket_process, ptr=None):
        super().__init__(ket_process=ket_process)

        self._as_parameter_ = API["ket_block_new"]() if ptr is None else ptr
        self._finalizer = weakref.finalize(
            self, API["ket_block_delete"], self._as_parameter_
        )

    def append_gate(self, gate, target):
        gate = json.dumps(gate).encode("utf-8")
        self.__getattr__("append_gate")(gate, target)

    def __getattr__(self, name: str):
        return lambda *args: API["ket_block_" + name](self, *args)

    def inverse(self):
        return Block(self.ket_process, self.__getattr__("inverse")())

    def control(self, qubits: list[int]):
        qubits_len = len(qubits)
        qubits = (qubits_len * c_size_t)(*qubits)
        return Block(
            self.ket_process,
            self.__getattr__("control")(qubits, qubits_len),
        )

    def take(self):
        ptr = self._as_parameter_
        self._finalizer.detach()

        self._as_parameter_ = None
        self._finalizer = None
        return ptr

    def set(self, other: Block):
        API["ket_block_delete"](self._as_parameter_)
        self._as_parameter_ = other.take()

    def __repr__(self) -> str:
        if self._as_parameter_ is not None:
            men_id = hex(id(self._as_parameter_))
        else:
            men_id = None
        return f"<Libket 'block', id={men_id}>"
