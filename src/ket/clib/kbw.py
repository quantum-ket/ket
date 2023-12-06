from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from ctypes import c_uint8, c_void_p, c_size_t, POINTER, c_int32
from os import environ
from os.path import dirname
from random import Random
import json
from .libket import JSON
from .wrapper import load_lib, from_u8_to_str, os_lib_name

DENSE = 0
SPARSE = 1

API_argtypes = {
    'kbw_run_and_set_result': ([c_void_p, c_int32], []),
    'kbw_run_serialized': ([POINTER(c_uint8), c_size_t, POINTER(c_uint8), c_size_t, c_int32, c_int32], [c_void_p]),   # pylint: disable=C0301
    'kbw_result_get': ([c_void_p], [POINTER(c_uint8), c_size_t]),
    'kbw_result_delete': ([c_void_p], []),
}


def kbw_path():
    """Get KBW path"""

    if "KBW_PATH" in environ:
        path = environ["KBW_PATH"]
    else:
        path = f'{dirname(__file__)}/libs/{os_lib_name("kbw")}'

    return path


API = load_lib('KBW', kbw_path(), API_argtypes, 'kbw_error_message')

SIM_MODE = None

RNG = Random()


def set_sim_mode_dense():
    """Set simulator mode to DENSE"""

    global SIM_MODE  # pylint: disable=W0603
    SIM_MODE = DENSE


def set_sim_mode_sparse():
    """Set simulator mode to SPARSE"""

    global SIM_MODE  # pylint: disable=W0603
    SIM_MODE = SPARSE


def set_seed(seed):
    """Initialize the simulator RNG"""

    global RNG  # pylint: disable=W0603
    RNG = Random(seed)


def set_dump_type(dump_type: str, shots: int | None = None):
    """Set the simulator dump type

    Args:
        dump_type: must be "vector", "probability", or "shots"
        shots: select the number of shots if ``dump_type`` is "shots"
    """

    if dump_type not in ["vector", "probability", "shots"]:
        raise ValueError('parameter "dump_type" must be "vector", "probability", or "shots"')

    environ['KBW_DUMP_TYPE'] = dump_type
    if shots is not None:
        environ['KBW_SHOTS'] = str(int(shots))


def run_and_set_result(process):
    """Execute quantum code from process"""

    environ['KBW_SEED'] = str(RNG.randint(0, (1 << 64) - 1))

    if SIM_MODE is None and 'KBW_MODE' in environ:
        sim_mode = environ['KBW_MODE'].upper()
        if sim_mode == 'DENSE':
            sim_mode = DENSE
        elif sim_mode == 'SPARSE':
            sim_mode = SPARSE
        else:
            raise RuntimeError(
                "undefined value for environment variable 'KBW_MODE', expecting 'DENSE' or 'SPARSE'")   # pylint: disable=C0301
    elif SIM_MODE is not None:
        sim_mode = SIM_MODE
    else:
        sim_mode = SPARSE

    API['kbw_run_and_set_result'](process, sim_mode)


def run_json(quantum_code, metrics, sim_mode: str = 'sparse'):
    """Call the simulator with a JSON quantum code"""

    quantum_code = json.dumps(quantum_code).encode()
    quantum_code_size = len(quantum_code)
    metrics = json.dumps(metrics).encode()
    metrics_size = len(metrics)

    sim_mode = {'SPARSE': SPARSE, 'DENSE': DENSE}[sim_mode.upper()]

    result = API['kbw_run_serialized'](
        (c_uint8 * quantum_code_size)(*quantum_code),
        quantum_code_size,
        (c_uint8 * metrics_size)(*metrics),
        metrics_size,
        JSON,
        sim_mode
    )

    data = json.loads(from_u8_to_str(*API['kbw_result_get'](result)))
    API['kbw_result_delete'](result)

    return data
