# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0
"""Ket Quantum Programming Platform.

Ket is a comprehensive library designed for quantum programming, providing essential functions and
classes to build quantum algorithms and applications. It facilitates the manipulation and storage of
quantum states, measurement operations, and the computation of expected values.

Explore the documentation of individual submodules for in-depth information and more practical code
examples.

All the functionality from the submodules is conveniently accessible within the ``ket`` namespace.
Except for the `lib` module.

Examples:

    - Grover algorithm

    .. code-block:: python

        from math import sqrt, pi
        import ket

        def grover(size: int, oracle) -> int:
            p = ket.Process(simulator="dense", num_qubits=size)

            s = ket.H(p.alloc(size)) 

            steps = int((pi / 4) * sqrt(2**size))  

            for _ in range(steps):
                oracle(s) 
                with ket.around(ket.H, s):
                    ket.phase_oracle(0, s)

            return ket.measure(s).value

    - Quantum teleportation protocol

    .. code-block:: python

        import ket

        def teleport(alice_msg, alice_aux, bob_aux):
            ket.ctrl(alice_msg, ket.X)(alice_aux)
            ket.H(alice_msg)

            m0 = ket.measure(alice_msg)
            m1 = ket.measure(alice_aux)

            if m1.value == 1:
                ket.X(bob_aux)
            if m0.value == 1:
                ket.Z(bob_aux)

            return bob_aux

        def bell(qubits):
            return ket.ctrl(ket.H(qubits[0]), ket.X)(qubits[1])

        def message(qubit):
            ket.H(alice)
            ket.Z(alice)

        p = ket.Process()

        alice = p.alloc()  # alice = |0⟩
        message(alice)     # alice = |–⟩

        bob = teleport(alice, *bell(p.alloc(2)))  # bob  <- alice
        
        ket.H(bob)         # bob   = |1⟩
        bob_m = ket.measure(bob)
        print("Expected measure 1, result =", bob_m.value)

"""


from .clib import libs
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
from . import lib

__version__ = "0.8.0rc1"

__all__ = all_base + all_func + all_gate + all_expv + all_state


def ket_version() -> list[str]:
    """Return the version of the Ket platform components."""
    from .clib.libket import API as libket  # pylint: disable=import-outside-toplevel
    from .clib.kbw import API as kbw  # pylint: disable=import-outside-toplevel

    libket_v, size = libket["ket_build_info"]()
    libket_v = bytearray(libket_v[: size.value])
    libket_v = libket_v.decode("utf-8")

    kbw_v, size = kbw["kbw_build_info"]()
    kbw_v = bytearray(kbw_v[: size.value])
    kbw_v = kbw_v.decode("utf-8")

    return [f"Ket v{__version__}", libket_v, kbw_v]
