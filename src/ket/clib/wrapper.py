# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Unitary for handle shared library with C API"""

from ctypes import POINTER, c_uint8, c_size_t, c_int32, cdll, c_char_p, byref
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


class LibketError(Exception):
    """Error from C libs"""

    def __init__(self, message, error_code):
        self.error_code = error_code
        self.message = message
        super().__init__(self.message)


class APIWrapper:  # pylint: disable=R0903
    """C API wrapper"""

    def __init__(self, lib_name, c_call, output, error_message, string_delete):
        self.lib_name = lib_name
        self.c_call = c_call
        self.output = output
        self.error_message = error_message
        self.string_delete = string_delete

    def __call__(self, *args):
        out = [c_type() for c_type in self.output]
        error_code = self.c_call(*args, *out)
        if error_code != 0:
            error_msg_ptr = c_char_p()
            if self.error_message(error_code, byref(error_msg_ptr)) == 0:
                error_msg = error_msg_ptr.value
                self.string_delete(error_msg_ptr)
                raise LibketError(f"Libket: {error_msg.decode('utf-8')}", error_code)
            else:
                raise LibketError("Libket: Unknown error", error_code)
        if len(out) == 1:
            return out[0]
        if len(out) != 0:
            return out
        return None


def load_lib(
    lib_name,
    lib_path,
    api_argtypes: dict[str, tuple[list, list]],
    error_message=None,
    string_delete=None,
):
    """Load clib"""

    lib = cdll.LoadLibrary(lib_path)
    if error_message is None:
        error_message = lib.__getattr__("ket_error_message")  # pylint: disable=C2801
        error_message.argtypes = [c_int32, POINTER(c_char_p)]
        error_message.restype = c_int32

    if string_delete is None:
        string_delete = lib.__getattr__("ket_string_delete")
        string_delete.argtypes = [c_char_p]
        string_delete.restype = c_int32

    api = {}
    for name, (args_in, args_out) in api_argtypes.items():
        c_call = lib.__getattr__(name)  # pylint: disable=C2801
        c_call.argtypes = [
            *args_in,
            *[POINTER(t) for t in args_out],
        ]

        api[name] = APIWrapper(
            lib_name,
            c_call,
            args_out,
            error_message,
            string_delete,
        )

    return api
