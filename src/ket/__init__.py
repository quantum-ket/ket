# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0
"""Ket quantum programming.

Every name below is available as ``ket.<name>`` (or ``from ket import *``),
except :mod:`ket.qulib`, the library of ready-made algorithms.

Quick tour:

.. code-block:: python

    from ket import *

    p = Process()          # runs on the KBW simulator
    q = p.alloc(2)         # a `Quant` with 2 qubits: q[0], q[1], q[:1], ...

    H(q[0])                # gates take qubits and return them
    with control(q[0]):    # every gate inside is controlled by q[0]
        X(q[1])

    print(sample(q).get())         # {0: ~1024, 3: ~1024}

    with obs():                    # build a Hamiltonian from Pauli operators
        h = Z(q[0]) * Z(q[1])
    print(exp_value(h).get())      # 1.0

    print(measure(q).value)        # 0 or 3, and the state collapses

Where to look next:

- :class:`~ket.base.Process`: simulators, ``live`` vs ``batch`` execution,
  and running on other backends.
- :mod:`ket.gates`: the gate set, and :func:`~ket.gates.obs` for observables.
- :mod:`ket.operations`: :func:`~ket.operations.control`,
  :func:`~ket.operations.adj`, :func:`~ket.operations.around`,
  :func:`~ket.operations.measure`, :func:`~ket.operations.sample`, and
  :func:`~ket.operations.exp_value`.
- :mod:`ket.qulib`: state preparation, oracles, Hamiltonians, and QUBO.
- :func:`set_log_level`: see what the runtime and simulator are doing.
"""

from importlib import metadata as _metadata

from .clib import libs
from .clib.log import set_log_level
from .base import *
from .base import __all__ as all_base
from .operations import *
from .operations import __all__ as all_func
from .gates import *
from .gates import __all__ as all_gate

from .expv import *
from .expv import __all__ as all_expv
from .quantumstate import *
from .quantumstate import __all__ as all_state
from .qint import *
from .qint import __all__ as all_qint
from .measurement import *
from .measurement import __all__ as all_measurement
from .fermion import *
from .fermion import __all__ as all_fermion
from . import qulib

try:
    __version__ = _metadata.version("ket-lang")
except _metadata.PackageNotFoundError:  # running from a source checkout
    __version__ = "unknown"

__all__ = (
    all_base
    + all_func
    + all_gate
    + all_expv
    + all_state
    + all_qint
    + all_measurement
    + all_fermion
    + ["qulib"]
    + ["set_log_level"]
)


def ket_version() -> list[str]:
    """Return the version strings of all Ket platform components.

    Queries the underlying Rust libraries (``libket`` and ``kbw``) for their build
    information and combines it with the Python package version.

    Returns:
        The versions of the Python package (``"Ket v<version>"``), of the
        ``libket`` runtime library, and of the ``kbw`` simulator, in this
        order.
    """
    from .clib.libket import (  # pylint: disable=import-outside-toplevel
        API as libket_api,
    )
    from .clib.kbw import API as kbw_api  # pylint: disable=import-outside-toplevel

    libket_v, size = libket_api["ket_build_info"]()
    libket_v = bytearray(libket_v[: size.value])
    libket_v = libket_v.decode("utf-8")

    kbw_v, size = kbw_api["kbw_build_info"]()
    kbw_v = bytearray(kbw_v[: size.value])
    kbw_v = kbw_v.decode("utf-8")

    return [f"Ket v{__version__}", libket_v, kbw_v]
