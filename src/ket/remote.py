"""Remote quantum execution module.

This module provides a tool to execute quantum code on a remote server. It includes 
the :class:`~ket.remote.Remote` class, which facilitates the connection to a remote
server.
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

import json
from ctypes import CFUNCTYPE, POINTER, c_uint8, c_size_t
from .clib.libket import BatchCExecution, make_configuration

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False


class Remote:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Remote quantum execution.

    The :class:`~ket.remote.Remote` class enables the execution of quantum circuits
    on a remote server. It provides functionality to establish a connection with
    the server and configure the connection parameters.

    The remote server's configuration is specified using the ``url`` parameter, while
    additional connection parameters can be passed as keyword arguments to the
    :meth:`~ket.remote.Remote.connect` method. The exact arguments required depend
    on the server's implementation. Refer to the server's documentation for detailed
    setup instructions.

    Once connected, the :class:`~ket.remote.Remote` instance integrates with a
    :class:`~ket.base.Process` object, which handles the actual quantum execution.
    The server determines the capabilities available for execution, such as
    calculating expectation values or dumping quantum states. The execution is set to
    batch.

    .. note::

        This class requires the `requests` library to be installed. You can install it with:
        ``pip install requests``

    Example:

        .. code-block:: python

            from ket.remote import Remote

            # Initialize a remote server connection
            remote = Remote("https://remote-server-url.com")

            # Connect to the remote server and start a quantum process
            process = Process(remote.connect(...))


    Args:
        url: The URL of the remote server.
        timeout: The timeout for the requests. Defaults to None.
    """

    def __init__(self, url: str, timeout: int | None = None):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests is not installed, run `pip install requests`")
        self._url = url
        self._logical_circuit = None
        self._physical_circuit = None
        self._result = None
        self._timeout = timeout
        self._result_json = bytearray()
        self._result_len = 0
        self._args = None

        @CFUNCTYPE(None, POINTER(c_uint8), c_size_t, POINTER(c_uint8), c_size_t)
        def submit_execution(
            logical_circuit,
            logical_circuit_size,
            physical_circuit,
            physical_circuit_size,
        ):
            self._logical_circuit = json.loads(
                bytearray(logical_circuit[:logical_circuit_size])
            )
            self._physical_circuit = json.loads(
                bytearray(physical_circuit[:physical_circuit_size])
            )
            self._submit()

        @CFUNCTYPE(None, POINTER(POINTER(c_uint8)), POINTER(c_size_t))
        def get_result(result_ptr, size):
            self._result_json = json.dumps(self._result).encode("utf-8")
            self._result_len = len(self._result_json)
            self._result_json = (c_uint8 * self._result_len)(*self._result_json)
            result_ptr[0] = self._result_json
            size[0] = self._result_len

        self.c_struct = BatchCExecution(
            submit_execution,
            get_result,
        )

    def _submit(self):
        url = f"{self._url}/run"
        response = requests.get(
            url,
            json=(self._logical_circuit, self._physical_circuit, self._args),
            timeout=self._timeout,
        )
        if response.status_code == 200:
            self._result = response.json()
        else:
            raise RuntimeError(f"Error: {response.status_code} - {response.text}")

    def connect(
        self,
        **kwargs,
    ):
        """
        Establishes a connection to the remote server.

        This method initializes a connection to the remote server using the provided
        keyword arguments. The returned configuration must be passed to the
        :class:`~ket.base.Process` constructor to create a new process connected to the server.

        Note that this method must be called for each new process that needs to
        interact with the server.

        Args:
            kwargs: Keyword arguments specifying connection parameters. The required
                    arguments depend on the remote server's API.

        Returns:
            Configuration object: The configuration required to initialize a process.
        """

        self._args = {k: str(v) for k, v in kwargs.items()}

        url = f"{self._url}/get"
        response = requests.get(
            url,
            json=self._args,
            timeout=self._timeout,
        )
        if response.status_code == 200:
            result = response.json()
        else:
            raise RuntimeError(f"Error: {response.status_code} - {response.text}")

        return make_configuration(batch_execution=self.c_struct, **result)
