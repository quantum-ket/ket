# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Unitary for handle shared library with C API"""

from ctypes import POINTER, c_int32, cdll, c_char_p, byref
import functools
import os
import threading

from .log import LogCallback, get_logger, install as install_logging

_LOGGER = get_logger("ket.clib")

# Error code returned by a Python callback that raised (`ExecutionFailed`).
CALLBACK_ERROR_CODE = 13

_callback_errors = threading.local()


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


def c_callback(func):
    """Make a Python function safe to call from Libket as a C callback.

    ctypes cannot pass a Python exception through C code: it only prints it,
    and the C side gets an undefined status code. With this decorator, an
    exception makes the callback return ``ExecutionFailed``, and the
    :class:`APIWrapper` call that led to the callback raises it again.
    The callback must return an ``int`` status code (``0`` for success).
    """

    @functools.wraps(func)
    def wrapper(*args):
        try:
            return func(*args)
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("callback %s raised %r", func.__name__, exc)
            if getattr(_callback_errors, "error", None) is None:
                _callback_errors.error = (func.__name__, exc)
            return CALLBACK_ERROR_CODE

    return wrapper


def _take_callback_error():
    error = getattr(_callback_errors, "error", None)
    _callback_errors.error = None
    return error


class APIWrapper:  # pylint: disable=R0903
    """C API wrapper"""

    def __init__(self, lib_name, c_call, output, error_message):
        self.lib_name = lib_name
        self.c_call = c_call
        self.output = output
        self.error_message = error_message

    def __call__(self, *args):
        out = [c_type() for c_type in self.output]
        _callback_errors.error = None
        error_code = self.c_call(*args, *out)
        if error_code != 0:
            callback_error = _take_callback_error()
            if callback_error is not None:
                # The error came from Python code called back by the library
                # (for example, a provider's `sample`): raise it as it was.
                name, exc = callback_error
                if hasattr(exc, "add_note"):
                    exc.add_note(f"(raised in the {name!r} callback of {self.lib_name})")
                raise exc
            # A short, static message (not freed); the library logged the
            # cause of the error to the `ket` logger.
            error_msg_ptr = c_char_p()
            if self.error_message(error_code, byref(error_msg_ptr)) == 0:
                error_msg = error_msg_ptr.value.decode("utf-8", "replace")
                raise LibketError(f"{self.lib_name}: {error_msg}", error_code)
            raise LibketError(f"{self.lib_name}: unknown error", error_code)
        if len(out) == 1:
            return out[0]
        if len(out) != 0:
            return out
        return None


def load_lib(
    lib_name,
    lib_path,
    api_argtypes: dict[str, tuple[list, list]],
    prefix: str = "ket",
    logger_name: str | None = None,
):
    """Load a Rust library with a C API.

    Args:
        lib_name: Name used in error messages.
        lib_path: Path of the shared library.
        api_argtypes: For each function, its input and output argument types.
        prefix: Prefix of the library's ``*_error_message`` and ``*_log_*``
            functions.
        logger_name: Logger that receives the library's log records.
    """

    lib = cdll.LoadLibrary(lib_path)
    error_message = getattr(lib, f"{prefix}_error_message")
    error_message.argtypes = [c_int32, POINTER(c_char_p)]
    error_message.restype = c_int32

    if logger_name is not None:
        set_callback = getattr(lib, f"{prefix}_log_set_callback")
        set_callback.argtypes = [LogCallback]
        set_callback.restype = c_int32
        set_level = getattr(lib, f"{prefix}_log_set_level")
        set_level.argtypes = [c_int32]
        set_level.restype = c_int32
        install_logging(logger_name, set_callback, set_level)

    api = {}
    for name, (args_in, args_out) in api_argtypes.items():
        c_call = getattr(lib, name)
        c_call.argtypes = [
            *args_in,
            *[POINTER(t) for t in args_out],
        ]

        api[name] = APIWrapper(
            lib_name,
            c_call,
            args_out,
            error_message,
        )

    return api
