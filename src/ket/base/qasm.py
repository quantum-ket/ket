# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

"""OpenQASM 2.0 integration for Ket.

This module provides the :func:`~ket.base.qasm.qasm` function, which allows
applying OpenQASM 2.0 circuits directly to quantum registers in a Ket process.
"""

from ctypes import c_size_t
from os import PathLike
from .quant import Quant
from ..clib.libket import Block, API as libket


def qasm(
    qubits: Quant,
    source: str | None = None,
    path: PathLike | None = None,
) -> Quant:
    """Apply an OpenQASM 2.0 circuit to a qubit register.

    Parses the supplied OpenQASM 2.0 program and appends the resulting gate
    sequence to the current quantum process.  The qubit register ``qubits``
    is mapped, in order, to the first ``qreg`` declared inside the QASM
    program.  Exactly one of ``source`` or ``path`` must be provided.

    Example:

        .. code-block:: python

            from ket import *

            p = Process()
            q = p.alloc(2)

            bell_qasm = \"\"\"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            h q[0];
            cx q[0],q[1];
            \"\"\"

            qasm(q, source=bell_qasm)
            print(dump(q).show())

        A file path can be used instead:

        .. code-block:: python

            qasm(q, path="bell.qasm")

    Args:
        qubits: The qubit register to apply the circuit to.  Its length must
            match the size of the first ``qreg`` declared in the QASM program.
        source: A string containing the OpenQASM 2.0 source code.
            Mutually exclusive with ``path``.
        path: Path to a ``.qasm`` file containing the OpenQASM 2.0 source
            code.  Mutually exclusive with ``source``.

    Returns:
        The input ``qubits`` register, unchanged, so that the call can be
        chained with other gate functions.

    Raises:
        TypeError: If ``qubits`` is not a :class:`~ket.base.Quant` instance.
        ValueError: If neither or both of ``source`` and ``path`` are given.
        FileNotFoundError: If ``path`` is provided but the file does not exist.
        OSError: If the file at ``path`` cannot be opened or read.
    """
    if not isinstance(qubits, Quant):
        raise TypeError(
            f"'qubits' must be a Quant instance, got {type(qubits).__name__!r}"
        )

    if source is None and path is None:
        raise ValueError(
            "one of 'source' or 'path' must be provided, but neither was given"
        )

    if source is not None and path is not None:
        raise ValueError(
            "only one of 'source' or 'path' may be provided, but both were given"
        )

    if source is not None:
        encoded = source.encode("utf-8")
    else:
        with open(path, "rb") as file:
            encoded = file.read()

    process = qubits.ket_process
    qubit_array = (c_size_t * len(qubits))(*qubits.qubits)
    block = Block(
        process,
        libket["ket_parse_openqasm"](qubit_array, len(qubits), encoded),
    )
    process.append_block(block)
    return qubits
