#  Copyright 2020, 2021 Evandro Chagas Ribeiro da Rosa <evandro.crr@posgrad.ufsc.br>
#  Copyright 2020, 2021 Rafael de Santiago <r.santiago@ufsc.br>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from ctypes import c_uint8, c_void_p, c_size_t, POINTER, c_int32
from os import environ
from os.path import dirname
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


def set_sim_mode_dense():
    """Set simulator mode to DENSE"""

    global SIM_MODE  # pylint: disable=W0603
    SIM_MODE = DENSE


def set_sim_mode_sparse():
    """Set simulator mode to SPARSE"""

    global SIM_MODE  # pylint: disable=W0603
    SIM_MODE = SPARSE


def run_and_set_result(process):
    """Execute quantum code from process"""

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
