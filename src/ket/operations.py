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

# pylint: disable=protected-access

from collections.abc import Sized
from contextlib import contextmanager
from functools import reduce, wraps
import json
from operator import add
from typing import Any, Callable
from inspect import signature
import warnings

from ket.clib.libket import search_process, API as libket


from .base import Process, Quant

from .measurement import Measurement, Samples
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
    "undo",
    "C",
    "kernel",
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
            qubits = reduce(Quant.__add__, qubits)

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

        process = qubits.ket_process

        for i, qubit in zip(state, qubits):
            if i == 0:
                with process.block_builder() as block:
                    block.append_gate("PauliX", qubit.qubits[0])
        return qubits

    if qubits is None:
        return inner
    return inner(qubits)


@contextmanager
def control(control_qubits: Quant, state: int | list[int] | None = None):
    r"""Controlled scope.

    Opens a controlled scope where quantum operations are applied only if the qubits of the
    parameter ``control_qubits`` are in the state :math:`\ket{1}`.

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
        state: **Deprecated.** The state to apply to the control qubits.
            Defaults to :math:`\ket{1}`.
    """

    if not isinstance(control_qubits, Quant):
        control_qubits = reduce(add, control_qubits)
    process = control_qubits.ket_process

    if state is not None:
        warnings.warn(
            "The 'state' parameter in 'control' is deprecated and will be"
            " removed in a future release."
            " Use with control('control_qubits' == 'state') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        with around(_flip_to_control(state), control_qubits, ket_process=process):
            with process.block_builder(control=control_qubits.qubits):
                try:
                    yield
                finally:
                    pass
    else:
        with process.block_builder(control=control_qubits.qubits):
            try:
                yield
            finally:
                pass


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
        ket_process = search_process(ket_process, args, kwargs)

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
    with process.block_builder(inverse=True):
        try:
            yield
        finally:
            pass


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
                f"Number of gates ({len(gates)}) is different from"
                " number of arguments ({len(args)})"
            )

        return tuple(
            gate(*arg) if isinstance(arg, tuple) else gate(arg)
            for gate, arg in zip(gates, args)
        )

    return inner


def undo(
    gate: Callable[[Quant], Any],
    qubits: Quant,
) -> Quant:
    """Automatic uncomputation of a quantum operation.

    Applies the specified ``gate`` on the ``qubits`` and returns a
    :class:`~ket.operations.Quant` instance. At the end of the returned
    object's lifecycle (when it is garbage-collected), the adjoint (inverse)
    of the gate is automatically applied to uncompute the operation.

    This is highly useful for managing temporary quantum states and ensuring
    they are properly disentangled before disposal.

    Args:
        gate: A callable representing the quantum gate/operation to apply.
        qubits: The quantum bits on which the gate will be applied.
    """

    ket_process = qubits.ket_process

    with ket_process.block_builder(append=False) as compute:
        gate(qubits)
        compute.lock_control()

    proprieties_ptr = compute.proprieties_json()
    proprieties = json.loads(proprieties_ptr.value)
    libket["ket_string_delete"](proprieties_ptr)

    blocked_qubits = set()

    for target, op in proprieties.items():
        target = int(target)
        if (op["propriety"] == "Permutation") and ket_process._is_aux(target):
            blocked_qubits.update(op["read_qubits"])
        elif (op["propriety"] == "Permutation") and ket_process._is_aux(target):
            raise RuntimeError("Operation not allowed on axillary qubit.")

    uncompute = compute.inverse()
    ket_process.append_block(compute, check_qubits=False)
    ket_process._block_qubits(blocked_qubits)

    def undo_func():
        if ket_process.status().value.decode("utf-8") == "Terminated":
            return

        ket_process._unblock_qubits(blocked_qubits)
        ket_process.append_block(uncompute, check_qubits=False)

    return Quant(
        qubits=qubits.qubits,
        process=ket_process,
        undo=undo_func,
        source=qubits,
    )


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

    ket_process = search_process(ket_process, args, kwargs)

    with ket_process.block_builder(append=False) as compute:
        gate(*args, **kwargs)
        compute.lock_control()

    proprieties_ptr = compute.proprieties_json()
    proprieties = json.loads(proprieties_ptr.value)
    libket["ket_string_delete"](proprieties_ptr)

    blocked_qubits = set()
    written_qubits = set()

    for target, op in proprieties.items():
        target = int(target)
        match op["propriety"]:
            case "Identity" | "Diagonal":
                pass
            case "Permutation":
                if ket_process._is_aux(target):
                    blocked_qubits.update(op["read_qubits"])
                written_qubits.add(target)
            case "Unitary":
                if ket_process._is_aux(target):
                    raise RuntimeError("Operation not allowed on axillary qubit.")
                written_qubits.add(target)

    allow_approximated_decomposition = True

    try:
        with ket_process.block_builder(append=False) as action:
            yield

        proprieties_ptr = action.proprieties_json()
        proprieties = json.loads(proprieties_ptr.value)
        libket["ket_string_delete"](proprieties_ptr)

        for target, op in proprieties.items():
            target = int(target)
            if op["propriety"] in ["Permutation", "Unitary"]:
                if target in blocked_qubits:
                    raise RuntimeError("Operation violates uncomputation.")
                if target in written_qubits:
                    allow_approximated_decomposition = False

    finally:

        if allow_approximated_decomposition:
            compute.enable_approximated_decomposition()

        uncompute = compute.inverse()
        ket_process.append_block(compute, check_qubits=False)
        ket_process.append_block(action, check_qubits=False)
        ket_process.append_block(uncompute, check_qubits=False)


def measure(qubits: Quant) -> Measurement:
    """Measure the given qubits and return a measurement object.

    Args:
        qubits: Qubits to be measured.

    Returns:
        Object representing the measurement results.
    """
    if not isinstance(qubits, Quant):
        qubits = reduce(Quant.__add__, qubits)

    postprocessing = (
        qubits.postprocessing() if hasattr(qubits, "postprocessing") else None
    )

    return Measurement(qubits, postprocessing)


def dump(*qubits: list[Quant]) -> QuantumState:
    """Obtain the quantum state snapshot.

    Args:
        qubits: Qubits to be observed.

    Returns:
        Object representing the quantum state.
    """

    return QuantumState(*qubits)


def sample(qubits: Quant, shots: int = 2048) -> Samples:
    """Get the quantum state measurement samples.

    Args:
        qubits: Qubits to be measured.
        shots: Number of measurement shots.

    Returns:
        Object representing the measurement samples.
    """
    if not isinstance(qubits, Quant):
        qubits = reduce(Quant.__add__, qubits)

    postprocessing = (
        qubits.postprocessing() if hasattr(qubits, "postprocessing") else None
    )

    return Samples(qubits, int(shots), postprocessing)


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
    """

    def inner(func):
        param = signature(func).parameters

        @wraps(func)
        def call(*args, ket_process: Process | None = None, **kwargs):
            ket_process = search_process(ket_process, args, kwargs)

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

                kwargs[name] = ket_process.alloc_aux(num_qubits)

            if unsafe:
                with ket_process.block_builder(diagonal=True):
                    return func(*args, **kwargs)
            return func(*args, **kwargs)

        return call

    return inner


def kernel(*p_args, **p_kwargs):
    """Quantum kernel decorator.

    Example:
        .. code-block:: python

            @kernel(num_qubits=2, execution="batch")(a=1, b=1)
            def bell(a, b):
                X(a + b)
                CNOT(H(a), b)

                with obs():
                    a0 = Z(a)
                    a1 = X(a)
                    b0 = -(X(b) + Z(b)) / sqrt(2)
                    b1 = (X(b) - Z(b)) / sqrt(2)
                    h = a0 * b0 + a0 * b1 + a1 * b0 - a1 * b1

                return exp_value(h).get()

            print(bell())
    """

    def make_process(**names):
        def make_kernel(func):
            param = signature(func).parameters

            @wraps(func)
            def call(*args, **kwargs):
                process = Process(*p_args, **p_kwargs)

                kwargs_ex = {**kwargs, **dict(zip(param, args))}

                for name, num_qubits in names.items():
                    if name in kwargs_ex:
                        continue

                    if callable(num_qubits):
                        try:
                            nq_args = {
                                p: kwargs_ex[p]
                                for p in signature(num_qubits).parameters
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

                    kwargs[name] = process.alloc(num_qubits)

                return func(*args, **kwargs)

            return call

        return make_kernel

    return make_process
