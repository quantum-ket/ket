"""Remote quantum execution module.

This module provides a tool to execute quantum code on a remote server. It includes
the :class:`~ket.remote.Remote` class, which facilitates the connection to a remote
server.
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0


from .clib.libket import BatchExecution

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False

try:
    import jwt
    from cryptography.hazmat.primitives import serialization
    import getpass
    from os import PathLike, path
    from datetime import datetime, timezone, timedelta

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False


class Remote(
    BatchExecution
):  # pylint: disable=too-few-public-methods,too-many-instance-attributes
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

        This module requires additional dependencies from ``ket-lang[remote]``.

        Install with: ``pip install ket-lang[remote]``.

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
        verify_ssl: Flag to disable SSL certificate verification.
            This value is passed directly to the ``requests.get()`` call.
            Defaults to :obj:`None`.
        private_key: Path to an OpenSSH RSA private key for authentication.
            If set to :obj:`None`, authentication will not be performed. Defaults to :obj:`None`.
        passphrase: Passphrase to decrypt the private key:

            - :obj:`True`: Prompts the user to enter the passphrase interactively;
            - :class:`bytes`: Provides the passphrase directly as a byte string;
            - :obj:`None`: Assumes the private key is unencrypted. Defaults to :obj:`None`.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        url: str,
        verify_ssl=None,
        timeout: int | None = None,
        private_key: PathLike | None = None,
        passphrase: bool | bytes | None = None,
    ):
        super().__init__()
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests is not installed, run `pip install requests`")
        self._url = url
        self._result = None
        self._timeout = timeout
        self._result_json = bytearray()
        self._result_len = 0
        self._args = None
        self._verify_ssl = verify_ssl

        if private_key is not None:
            if not JWT_AVAILABLE:
                raise ImportError(
                    "jwt is not installed, run `pip install PyJWT[crypto] bcrypt`"
                )

            if isinstance(passphrase, bool):
                if passphrase:
                    passphrase = bytes(
                        getpass.getpass(f"passphrase for {private_key}:"), "utf-8"
                    )
                else:
                    passphrase = None

            with open(path.expandvars(path.expanduser(private_key)), "rb") as key_file:
                private_key = serialization.load_ssh_private_key(
                    key_file.read(), password=passphrase
                )
            self._token = jwt.encode(
                {
                    "iss": "org.quantumket.ket",
                    "exp": datetime.now(tz=timezone.utc)
                    + timedelta(seconds=timeout if timeout is not None else 604800),
                },
                private_key,
                algorithm="PS256",
            )
        else:
            self._token = None

    def submit_execution(
        self,
        logical_circuit: dict,
        physical_circuit: dict | None,
        parameters: list[float],
    ):
        url = f"{self._url}/run"
        response = requests.get(
            url,
            verify=self._verify_ssl,
            json=(logical_circuit, physical_circuit, parameters, self._args),
            timeout=self._timeout,
        )
        if response.status_code == 200:
            self._result = response.json()
        else:
            raise RuntimeError(f"Error: {response.status_code} - {response.text}")

    def get_result(self) -> dict:
        return self._result

    def clear(self):
        pass

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
        if self._token is not None:
            self._args["token"] = self._token

        url = f"{self._url}/get"
        response = requests.get(
            url,
            verify=self._verify_ssl,
            json=self._args,
            timeout=self._timeout,
        )
        if response.status_code == 200:
            result = response.json()
        else:
            raise RuntimeError(f"Error: {response.status_code} - {response.text}")

        return super().configure(**result)
