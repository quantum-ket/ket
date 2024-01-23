# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Unitary for handle shared library with C API"""

from ctypes import POINTER, c_uint8, c_size_t, c_int32, cdll
import os


def os_lib_name(lib):
    """Append the OS specific extensions to a lib name"""

    if os.name == "nt":
        return f"{lib}.dll"
    if os.uname().sysname == "Linux":
        return f"lib{lib}.so"
    if os.uname().sysname == "Darwin":
        return f"lib{lib}.dylib"
    raise OSError("unsupported operational system")


def from_u8_to_str(data, size):
    """Convert a unsigned char vector to a Python string"""

    return bytearray(data[: size.value]).decode()


class CLibError(Exception):
    """Error from C libs"""

    def __init__(self, message, error_code):
        self.error_code = error_code
        self.message = message
        super().__init__(self.message)


class APIWrapper:  # pylint: disable=R0903
    """C API wrapper"""

    def __init__(self, lib_name, c_call, output, error_message):
        self.lib_name = lib_name
        self.c_call = c_call
        self.output = output
        self.error_message = error_message

    def __call__(self, *args):
        out = [c_type() for c_type in self.output]
        error_code = self.c_call(*args, *out)
        if error_code != 0:
            error_msg_buffer_size = 128
            error_message_buffer = (c_uint8 * error_msg_buffer_size)()
            write_size = c_size_t()
            while (
                self.error_message(
                    error_code, error_message_buffer, error_msg_buffer_size, write_size
                )
                != 0
            ):
                error_msg_buffer_size = write_size.value
                error_message_buffer = (c_uint8 * error_msg_buffer_size)()

            error_msg = bytearray(error_message_buffer[: write_size.value]).decode()
            raise CLibError(f"{self.lib_name}: {error_msg}", error_code)
        if len(out) == 1:
            return out[0]
        if len(out) != 0:
            return out
        return None


def load_lib(lib_name, lib_path, api_argtypes, error_message):
    """Load clib"""

    lib = cdll.LoadLibrary(lib_path)
    error_message = lib.__getattr__(error_message)  # pylint: disable=C2801
    error_message.argtypes = [c_int32, POINTER(c_uint8), c_size_t, POINTER(c_size_t)]
    error_message.restype = c_int32

    api = {}
    for name in api_argtypes:
        c_call = lib.__getattr__(name)  # pylint: disable=C2801
        c_call.argtypes = [
            *api_argtypes[name][0],
            *[POINTER(t) for t in api_argtypes[name][1]],
        ]

        api[name] = APIWrapper(lib_name, c_call, api_argtypes[name][1], error_message)

    return api
