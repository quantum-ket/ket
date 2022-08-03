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

from ctypes import *


def from_u8_to_str(data, size):
    return bytearray(data[:size.value]).decode()


def from_list_to_c_vector(data):
    return (c_void_p*len(data))(*(d._as_parameter_ for d in data)), len(data)


class clib_error(Exception):
    def __init__(self, message, error_code):
        self.error_code = error_code
        self.message = message
        super().__init__(self.message)


class APIWrapper:
    def __init__(self, lib_name, c_call, output, error_message):
        self.lib_name = lib_name
        self.c_call = c_call
        self.output = output
        self.error_message = error_message

    def __call__(self, *args):
        out = [c_type() for c_type in self.output]
        error_code = self.c_call(*args, *out)
        if error_code != 0:
            size = c_size_t()
            error_msg = self.error_message(error_code, size)
            raise clib_error(
                self.lib_name+': '+from_u8_to_str(error_msg, size), error_code)
        elif len(out) == 1:
            return out[0]
        elif len(out) != 0:
            return out


def load_lib(lib_name, lib_path, API_argtypes, error_message):

    lib = cdll.LoadLibrary(lib_path)
    error_message = lib.__getattr__(error_message)
    error_message.argtypes = [c_int32, POINTER(c_size_t)]
    error_message.restype = POINTER(c_uint8)

    API = {}
    for name in API_argtypes:
        c_call = lib.__getattr__(name)
        c_call.argtypes = [
            *API_argtypes[name][0],
            *[POINTER(t) for t in API_argtypes[name][1]]
        ]

        API[name] = APIWrapper(
            lib_name,
            c_call,
            API_argtypes[name][1],
            error_message
        )

    return API
