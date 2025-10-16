"""Functions to manipulate quantum operations.

This module provides essential functions in the Ket library for manipulating quantum operations.
It includes functions for controlled and inverse operations, facilitating quantum circuit
construction.
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Sized
from contextlib import contextmanager
from ctypes import c_size_t
from functools import reduce, wraps
from operator import add
from typing import Any, Callable, Sequence
from inspect import signature

from ket.clib.libket import PAULI_X


from .base import (
    Process,
    Quant,
    Measurement,
    Samples,
)
from .quantumstate import QuantumState


from .expv import (
    Pauli,
    Hamiltonian,
    ExpValue,
)

__all__ = [
    "control",
    "ctrl",
    "inverse",
    "adj",
    "around",
    "cat",
    "kron",
    "measure",
    "sample",
    "dump",
    "exp_value",
    "using_aux",
    "is_permutation",
    "is_diagonal",
    "C",
]


def _flip_to_control(
    control_state: int | list[int] | None, qubits: Quant | None = None
) -> Quant | Callable[[Quant], Quant]:
    r"""Flip qubits from :math:`\ket{\texttt{control_state}}` to :math:`\ket{1\dots1}`.

    The primary usage of this gate is to change the state when controlled applications are applied.
    For instance, all controlled operations are only applied if the control qubits' state is
    :math:`\ket{1}`. This gate is useful for using another state as control.

    Example:

        .. code-block:: python

            from ket import *

            p = Process()
            q = p.alloc(3)

            H(q[:2])

            with around(_flip_to_control(0b01), q[:2]):
                ctrl(q[:2], X)(q[2])

    Args:
        control_state: The state to flip the control qubits to.
        qubits: The qubits to apply the flip to.
    """

    def inner(qubits: Quant) -> Quant:
        if control_state is None:
            return qubits

        if not isinstance(qubits, Quant):
            qubits = reduce(add, qubits)

        length = len(qubits)
        if isinstance(control_state, Sized):
            if len(control_state) != length:
                raise ValueError(
                    f"'to' received a list of length {len(control_state)} to use on {length} qubits"
                )
            state = control_state
        else:
            if length < control_state.bit_length():
                raise ValueError(
                    f"To flip with control_state={control_state} "
                    f"you need at least {control_state.bit_length()} qubits"
                )

            state = [int(i) for i in f"{{:0{length}b}}".format(control_state)]

        for i, qubit in zip(state, qubits):
            if i == 0:
                qubit.process.apply_gate(PAULI_X, 0.0, False, 0, qubit.qubits[0])
        return qubits

    if qubits is None:
        return inner
    return inner(qubits)


@contextmanager
def control(control_qubits: Quant, state: int | list[int] | None = None):
    r"""Controlled scope.

    Opens a controlled scope where quantum operations are applied only if the qubits of the
    parameter ``control_qubits`` are in the state :math:`\ket{1}`. The parameter ``state`` can
    be used to specify a different state.

    :Usage:

    .. code-block:: python

        with control(control_qubits):
            # Apply operations only if control_qubits are in the state |1>
            ...

    Example:

        .. code-block:: python

            from ket import *

            p = Process()

            c = p.alloc(2)
            a, b = q.alloc(2)

            # CNOT c[0] a
            with control(c[0]):
                X(a)

            # Toffoli c[0] c[1] a
            with control(c):
                X(a)

            # CSWAP c[0] a b
            with control(c[0]):
                SWAP(a, b)

    Args:
        control_qubits: The qubits to control the quantum operations.
        state: The state to apply to the control qubits. Defaults to :math:`\ket{1}`.
    """

    if not isinstance(control_qubits, Quant):
        control_qubits = reduce(add, control_qubits)
    process = control_qubits.process

    if state is not None:
        with around(_flip_to_control(state), control_qubits, ket_process=process):
            process.ctrl_push(
                (c_size_t * len(control_qubits.qubits))(*control_qubits.qubits),
                len(control_qubits.qubits),
            )
            try:
                yield
            finally:
                process.ctrl_pop()
    else:
        process.ctrl_push(
            (c_size_t * len(control_qubits.qubits))(*control_qubits.qubits),
            len(control_qubits.qubits),
        )
        try:
            yield
        finally:
            process.ctrl_pop()


def ctrl(
    control_qubits: Quant, gate: Callable, state: int | list[int] | None = None
) -> Callable:
    """Add control qubits to a gate.

    Create a new callable that applies the given ``gate`` with the control qubits.

    :Usage:

    .. code-block:: python

        from ket import *

        p = Process()

        c = p.alloc(2)
        a, b = q.alloc(2)

        # CNOT c[0] a
        ctrl(c[0], X)(a)

        # Toffoli c[0] c[1] a
        ctrl(c, X)(a)

        # CSWAP c[0] a b
        ctrl(c[0], SWAP)(a, b)

    Args:
        control_qubits: The qubits to control the quantum operations.
        gate: The gate to apply with the control qubits.

    Returns:
        A new callable that applies the given gate with the control qubits.
    """

    @wraps(gate)
    def inner(*args, **kwargs):
        with control(control_qubits, state):
            return control_qubits, gate(*args, **kwargs)

    return inner


def C(gate: Callable) -> Callable:  # pylint: disable=invalid-name
    """Create a controlled gate

    Args:
        gate: Quantum gate.

    Returns:
        A gate with the first parameter as control.
    """

    @wraps(gate)
    def inner(control_qubits, *args, **kwargs):
        return ctrl(control_qubits, gate)(*args, **kwargs)

    return inner


def _search_process(ket_process, args, kwargs):
    def inner(ket_process, arg):
        if hasattr(arg, "_get_ket_process"):
            arg_process = arg._get_ket_process()  # pylint: disable=protected-access
            if ket_process is not None and ket_process is not arg_process:
                raise ValueError("parameter with different Ket processes")
            ket_process = arg_process
        return ket_process

    def search(ket_process, args):
        for arg in args:
            if isinstance(arg, Sequence) and not isinstance(arg, str):
                for subarg in arg:
                    ket_process = inner(ket_process, subarg)
            else:
                ket_process = inner(ket_process, arg)
        return ket_process

    if ket_process is None:
        ket_process = search(ket_process, args)
        ket_process = search(ket_process, kwargs.values())

    if ket_process is None:
        raise ValueError("Ket process not found in the parameters")

    return ket_process


def adj(gate: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Return the inverse of a gate.

    Create a new callable that applies the inverse of the given ``gate``.

    The resulting callable will iterate over the parameters to identify the
    :class:`~ket.base.Process` that the inverse operation will be applied. If non of the parameters
    are a :class:`~ket.base.Quant`, the keyword argument ``ket_process`` must be given.

    :Usage:

    .. code-block:: python

        from ket import *

        p = Process()

        a, b = q.alloc(2)

        bell = cat(kron(H, I), CNOT)

        adj(bell)(a, b)

    Args:
        gate: The gate to apply the inverse.

    Returns:
        A new callable that applies the inverse of the given gate.

    """

    @wraps(gate)
    def inner(*args, ket_process: Process | None = None, **kwargs) -> Any:
        ket_process = _search_process(ket_process, args, kwargs)

        with inverse(ket_process):
            return gate(*args, **kwargs)

    return inner


@contextmanager
def inverse(process: Process):
    """Inverse scope.

    Open a scope where all the quantum operations are executed backwards.

    :Usage:

    .. code-block:: python

       with inverse(process):
            # Classical and quantum operations
            ...

    Args:
        process: The process where the inverse operations will be applied.
    """
    process.adj_begin()
    try:
        yield
    finally:
        process.adj_end()


def cat(*gates) -> Callable[[Any], Any]:
    """Concatenate gates.

    Create a new callable concatenating all the quantum gates.

    Example:

        .. code-block:: python

            from ket import *

            z_gate = cat(H, X, H)

            p = process()
            q = p.alloc()

            z_gate(q)

    Args:
        *gates: Quantum gates to concatenate.

    Returns:
        A new callable that concatenates the given quantum gates.
    """

    def inner(*args):
        for gate in gates:
            gate(*args)

        if len(args) == 1:
            return args[0]
        return args

    return inner


def kron(*gates, n: int = 1) -> Callable[[Any], Any]:
    """Gates tensor product.

    Create a new callable that with the tensor product of the given gates.

    Example:

        .. code-block:: python

            from ket import *

            p = process()
            a, b = p.alloc(2)

            HX = kron(H, X)

            HX(a, b) # Apply an Hadamard on qubit `a` and a Pauli X on qubit `b`

    Args:
        *gates: Quantum gates to tensor product.

    Returns:
        A new callable that represents the tensor product of the given quantum gates.
    """

    gates = gates * n

    def inner(*args):
        if len(gates) != len(args):
            raise ValueError(
                f"Number of gates ({len(gates)}) is different from number of arguments ({len(args)})"
            )

        return tuple(
            gate(*arg) if isinstance(arg, tuple) else gate(arg)
            for gate, arg in zip(gates, args)
        )

    return inner


@contextmanager
def around(gate: Callable, *args, ket_process: Process | None = None, **kwargs):
    r"""Applying and then reversing quantum gates.

    Apply the given quantum gate (:math:`U`) and then execute a code block (:math:`V`). After the
    code block is executed, the inverse of the gate (:math:`U^\dagger`) is applied, resulting in
    :math:`UVU^\dagger`.

    Example:

        .. code-block:: python

            from ket import *

            p = Process()
            a, b = p.alloc(2)

            bell = cat(kron(H, I), CNOT)

            with around(bell, a, b): # Apply bell(a, b)
                X(q)                 # Execute the code block
                                    # Apply adj(bell)(a, b)

    Args:
        gate: Quantum gate to apply.
        *arg: Qubits or classical parameters for the gate.
        ket_process: Explicitly specify the quantum process. If not provided, the process
                     is inferred from the qubits.
        **kwargs: Additional parameters for the quantum gate.

    """

    ket_process = _search_process(ket_process, args, kwargs)

    ket_process.around_begin()

    ket_process.ctrl_stack()
    # ket_process.approximated_decomposition_begin()
    gate(*args, **kwargs)
    # ket_process.approximated_decomposition_end()
    ket_process.ctrl_unstack()

    ket_process.around_mid()

    try:
        yield
    finally:

        ket_process.around_undo()

        ket_process.ctrl_stack()
        # ket_process.approximated_decomposition_begin()
        adj(gate)(*args, ket_process=ket_process, **kwargs)
        # ket_process.approximated_decomposition_end()
        ket_process.ctrl_unstack()

        ket_process.around_end()


def measure(qubits: Quant) -> Measurement:
    """Measure the given qubits and return a measurement object.

    Args:
        qubits: Qubits to be measured.

    Returns:
        Object representing the measurement results.
    """
    if not isinstance(qubits, Quant):
        qubits = reduce(add, qubits)

    return Measurement(qubits)


def dump(qubits: Quant) -> QuantumState:
    """Obtain the quantum state snapshot.

    Args:
        qubits: Qubits to be observed.

    Returns:
        Object representing the quantum state.
    """
    if not isinstance(qubits, Quant):
        qubits = reduce(add, qubits)

    return QuantumState(qubits)


def sample(qubits: Quant, shots: int = 2048) -> Samples:
    """Get the quantum state measurement samples.

    Args:
        qubits: Qubits to be measured.
        shots: Number of measurement shots.

    Returns:
        Object representing the measurement samples.
    """
    if not isinstance(qubits, Quant):
        qubits = reduce(add, qubits)

    return Samples(qubits, int(shots))


def exp_value(hamiltonian: Hamiltonian | Pauli) -> ExpValue:
    """Calculate the expected value for a quantum state.

    Args:
        hamiltonian: Hamiltonian or Pauli operator for calculating the expected value.

    Returns:
        Object representing the expected value.
    """
    return ExpValue(hamiltonian)


def using_aux(unsafe: bool = False, **names):
    """Add axillary qubits to a quantum gate.

    Example:
            .. code-block:: python

                @using_aux(a=lambda c: 0 if len(c) <= 2 else 1)
                def v_chain(c, t, a):
                    if len(c) <= 2:
                        ctrl(c, X)(t)
                    else:
                        with around(ctrl(c[:2], X), a):
                            v_chain(a + c[2:], t)

    Args:
        unsafe: Use dirty unsafe qubits. Defaults to False.
    """

    def inner(func):
        param = signature(func).parameters

        @wraps(func)
        def call(*args, ket_process: Process | None = None, **kwargs):
            ket_process = _search_process(ket_process, args, kwargs)

            if unsafe:
                q_args = [q for q in args if isinstance(q, Quant)] + [
                    q for q in kwargs.values() if isinstance(q, Quant)
                ]
                interacting_qubits = reduce(add, q_args)
                interacting_qubits = interacting_qubits.qubits
                interacting_qubits_size = len(interacting_qubits)
                interacting_qubits = (c_size_t * interacting_qubits_size)(
                    *interacting_qubits
                )
            else:
                interacting_qubits_size = 0
                interacting_qubits = None

            free_stack = []

            kwargs_ex = {**kwargs, **dict(zip(param, args))}

            for name, num_qubits in names.items():
                if name in kwargs_ex:
                    continue

                if callable(num_qubits):
                    try:
                        nq_args = {
                            p: kwargs_ex[p] for p in signature(num_qubits).parameters
                        }
                        num_qubits = num_qubits(**nq_args)
                    except KeyError as e:
                        raise ValueError(
                            "Parameters used to calculate the number of auxiliary "
                            "qubit must be passed as keyword arguments"
                        ) from e

                if num_qubits <= 0:
                    kwargs[name] = None
                    continue

                aux_index, aux_id = ket_process.allocate_aux(
                    num_qubits, interacting_qubits, interacting_qubits_size
                )

                qubits = Quant(
                    qubits=[
                        (i + 1) << 32
                        for i in range(aux_index.value, aux_index.value + num_qubits)
                    ],
                    process=ket_process,
                )

                free_stack.append(aux_id.value)

                kwargs[name] = qubits

            func(*args, **kwargs)

            for aux_id in reversed(free_stack):
                ket_process.free_aux(aux_id)

        return call

    return inner


def is_diagonal(gate: Callable) -> Callable:
    """Force to consider as a diagonal gate.

    Args:
        gate: Quantum gate
    """

    @wraps(gate)
    def inner(*args, ket_process: Process | None = None, **kwargs) -> Any:
        ket_process = _search_process(ket_process, args, kwargs)

        ket_process.is_diagonal_begin()
        try:
            return gate(*args, **kwargs)
        finally:
            ket_process.is_diagonal_end()

    return inner


def is_permutation(gate: Callable) -> Callable:
    """Force to consider as a permutation gate.

    Args:
        gate: Quantum gate
    """

    @wraps(gate)
    def inner(*args, ket_process: Process | None = None, **kwargs) -> Any:
        ket_process = _search_process(ket_process, args, kwargs)

        ket_process.is_permutation_begin()
        try:
            return gate(*args, **kwargs)
        finally:
            ket_process.is_permutation_end()

    return inner
