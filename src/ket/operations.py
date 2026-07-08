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
    r"""Context manager that conditions all operations in its body on a qubit state.

    Opens a controlled scope where every quantum operation applied inside is
    conditioned on the ``control_qubits`` being in the :math:`\left|1\right\rangle`
    state (by default). All operations in the body are automatically wrapped in
    a multi-qubit controlled block.

    :Usage:

        .. code-block:: python

            with control(control_qubits):
                # All operations here are applied only if control_qubits = |1⟩
                ...

    Example:

        .. code-block:: python

            from ket import *
            p = Process()
            c = p.alloc(2)
            a, b = p.alloc(2)
            # CNOT: flip `a` if c[0] = |1⟩
            with control(c[0]):
                X(a)
            # Toffoli: flip `a` if both c[0] = c[1] = |1⟩
            with control(c):
                X(a)
            # Fredkin (CSWAP): swap `a` and `b` if c[0] = |1⟩
            with control(c[0]):
                SWAP(a, b)

    Args:
        control_qubits: The qubit(s) that act as
            controls. All qubits must belong to the same process.
        state: **Deprecated.** Bit pattern specifying
            the control state. Use ``with control(q == state):`` instead.
            Defaults to :math:`\left|1\right\rangle`.

    .. deprecated:: 0.9.3
        The ``state`` argument is deprecated. Replace::

            with control(q, state=0b01):
                ...

        with::

            with control(q == 0b01):
                ...
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
    """Convert a gate into a controlled version where the first argument is the control.

    This is a convenience decorator/wrapper that reorders the arguments of a
    gate so that the first positional argument becomes the control qubit(s).
    It is equivalent to calling ``ctrl(control_qubits, gate)(*args)``.

    Example:

        .. code-block:: python

            from ket import *
            p = Process()
            c, a, b = p.alloc(3)
            CX = C(X)          # CX is equivalent to CNOT
            CCNOT = C(C(X))    # Toffoli gate
            CX(c, a)           # CNOT: c is control, a is target
            CCNOT(c, a, b)     # Toffoli: c and a are controls, b is target

    Args:
        gate: The quantum gate to make controlled.

    Returns:
        A new callable where the first positional argument is used as
        the control qubit(s) for ``gate``.
    """

    @wraps(gate)
    def inner(control_qubits, *args, **kwargs):
        return ctrl(control_qubits, gate)(*args, **kwargs)

    return inner


def adj(gate: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Return the adjoint (inverse/Hermitian conjugate) of a gate.

    Creates a new callable that applies the time-reversed sequence of
    operations of the given ``gate``. For unitary gates, the adjoint is
    the inverse operation.

    The process is inferred automatically from the :class:`~ket.base.Quant`
    arguments. If no :class:`~ket.base.Quant` is present in the arguments
    (e.g., when all parameters are classical), the keyword argument
    ``ket_process`` must be supplied explicitly.

    :Usage:

        .. code-block:: python

            from ket import *
            p = Process()
            a, b = p.alloc(2)
            # Prepare a Bell state and then un-prepare it
            bell = cat(kron(H, I), CNOT)
            bell(a, b)          # entangle
            adj(bell)(a, b)     # un-entangle: returns |00⟩

    Args:
        gate: The quantum gate to invert.

    Returns:
        A new callable that applies the adjoint of ``gate``.
        Accepts the same arguments as ``gate`` plus an optional
        ``ket_process`` keyword argument.
    """

    @wraps(gate)
    def inner(*args, ket_process: Process | None = None, **kwargs) -> Any:
        ket_process = search_process(ket_process, args, kwargs)

        with inverse(ket_process):
            return gate(*args, **kwargs)

    return inner


@contextmanager
def inverse(process: Process):
    """Context manager that reverses the order of all quantum operations in its body.

    All gates applied inside the ``with inverse(process):`` block are
    collected and then appended to the circuit in **reverse order**, with each
    individual gate also inverted (i.e., the adjoint of each gate). This
    implements the unitary inverse :math:`U^\\dagger` of the sequence
    :math:`U`.

    :Usage:

        .. code-block:: python

            with inverse(process):
                # Operations here are appended in reversed, adjoint form
                ...

    Example:

        .. code-block:: python

            from ket import *
            p = Process()
            q = p.alloc(2)
            H(q[0])
            CNOT(q[0], q[1])      # Prepare Bell state
            with inverse(p):      # Undo the Bell preparation
                CNOT(q[0], q[1])
                H(q[0])
            # q is now back to |00⟩

    Args:
        process: The process to apply the inverse
            scope to.
    """
    with process.block_builder(inverse=True):
        try:
            yield
        finally:
            pass


def cat(*gates) -> Callable[[Any], Any]:
    """Create a sequential composition (concatenation) of quantum gates.

    Returns a new callable that applies all the given ``gates`` one after the
    other to the **same** set of arguments. This is the quantum analogue of
    function composition when all gates operate on the same qubits.

    Gate application order matches the argument order: ``cat(U, V)`` first
    applies ``U``, then ``V`` (left-to-right).

    Example:

        .. code-block:: python

            from ket import *
            # A Z gate can be decomposed as H-X-H
            z_gate = cat(H, X, H)
            p = Process()
            q = p.alloc()
            z_gate(q)     # equivalent to Z(q)

    Args:
        *gates: Quantum gate callables to compose sequentially.
            Each gate receives the full argument list.

    Returns:
        A single callable that applies all ``gates`` in order.
        If only one argument was passed to the composed callable, returns that
        argument; if multiple were passed, returns them as a tuple.
    """

    def inner(*args):
        for gate in gates:
            gate(*args)

        if len(args) == 1:
            return args[0]
        return args

    return inner


def kron(*gates, n: int = 1) -> Callable[[Any], Any]:
    """Create a tensor-product (parallel) composition of quantum gates.

    Returns a new callable that applies each gate to the corresponding
    positional argument independently. Unlike :func:`~ket.operations.cat`,
    which applies all gates to the *same* arguments, ``kron`` maps
    each gate to a *separate* argument, mirroring the tensor product
    :math:`U_1 \\otimes U_2 \\otimes \\cdots`.

    The optional ``n`` parameter repeats the entire gate list ``n`` times,
    which is useful for applying the same set of gates across multiple
    register pairs.

    Example:

        .. code-block:: python

            from ket import *
            p = Process()
            a, b, c, d = p.alloc(4)
            HX = kron(H, X)
            HX(a, b)   # H on a, X on b
            # Apply H⊗H⊗H to three separate qubits at once
            HHH = kron(H, n=3)
            HHH(a, b, c)   # H on each

    Args:
        *gates: Quantum gate callables to apply in parallel. Gate ``i`` is
            applied to argument ``i``.
        n: Repeat the full gate list ``n`` times. Defaults to ``1``.

    Returns:
        A new callable that applies each gate to its corresponding
        argument and returns all results as a tuple.

    Raises:
        ValueError: If the number of gates does not match the number of
            arguments supplied when the resulting callable is called.
    """

    gates = gates * n

    def inner(*args):
        if len(gates) != len(args):
            raise ValueError(
                f"Number of gates ({len(gates)}) is different from"
                f" number of arguments ({len(args)})"
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
    """Apply a gate and schedule its adjoint for automatic uncomputation.

    Applies the specified ``gate`` to ``qubits`` immediately and returns a
    new :class:`~ket.base.Quant` wrapping the same qubits. When this returned
    object is garbage-collected or goes out of scope (or is used as a context
    manager and exits), the **adjoint** of ``gate`` is automatically appended
    to the circuit, uncomputing the operation.

    This is the recommended pattern for managing temporary quantum states:
    apply a computation, use its result, and rely on automatic cleanup rather
    than manually calling :func:`~ket.operations.adj`.

    Example:

        .. code-block:: python

            from ket import *
            p = Process()
            c, t = p.alloc(2)
            H(c)
            # Compute a CNOT result, then auto-uncompute on scope exit
            with undo(ctrl(c, X), t) as flipped:
                sample_result = sample(c + flipped)
            # adjoint CNOT is automatically applied here

    Args:
        gate: The quantum gate or operation to apply.
            Must accept a :class:`~ket.base.Quant` as its argument.
        qubits: The target qubits for the gate.

    Returns:
        A :class:`~ket.base.Quant` wrapping the same
        qubit indices. When this object is finalized, the adjoint of ``gate``
        is appended to the circuit.

    Raises:
        RuntimeError: If ``gate`` attempts a non-permutation operation on an
            auxiliary qubit, which would violate uncomputation safety.
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
        elif (op["propriety"] == "Unitary") and ket_process._is_aux(target):
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
    r"""Apply a gate, execute a body block, then apply the gate's adjoint: :math:`UVU^\dagger`.

    This context manager implements the *conjugation pattern* common in quantum
    algorithms. It applies the given gate :math:`U`, then runs the code inside
    the ``with`` block (:math:`V`), and finally automatically appends
    :math:`U^\dagger` (the adjoint of :math:`U`). The resulting circuit
    fragment is :math:`UVU^\dagger`.

    This is particularly useful for changing the basis of an operation without
    manually writing the inverse:

    Example:

        .. code-block:: python

            from ket import *
            p = Process()
            a, b = p.alloc(2)
            # CZ gate expressed as H-CNOT-H (change of basis)
            with around(H, b):
                CNOT(a, b)  # CNOT in the Z basis becomes CZ in the X basis

    Example with multi-qubit gate:

    .. code-block:: python

        from ket import *
        p = Process()
        a, b = p.alloc(2)
        bell = cat(kron(H, I), CNOT)
        with around(bell, a, b):  # Apply bell(a, b) = U
            X(a)                  # V is applied in the Bell basis
                                  # adj(bell)(a, b) = U† is applied on exit

    Args:
        gate: The quantum gate :math:`U` to apply before and invert
            after the body.
        *args: Positional arguments forwarded to ``gate``.
        ket_process: Explicitly specify the
            quantum process. If ``None``, the process is inferred automatically
            from the qubit arguments.
        **kwargs: Additional keyword arguments forwarded to ``gate``.

    Raises:
        RuntimeError: If the body block attempts an operation that violates
            uncomputation rules (e.g., writing to a blocked auxiliary qubit).
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
    """Measure qubits in the computational basis and return a measurement handle.

    Schedules a measurement operation on the given ``qubits``.

    The measured integer represents the bit-string of the qubits in big-endian
    order: the first qubit in the register is the most significant bit.

    Example:

        .. code-block:: python

            from ket import *
            p = Process()
            q = p.alloc(2)
            CNOT(H(q[0]), q[1])   # Bell state
            result = measure(q)
            print(result.value)   # 0 or 3 (|00⟩ or |11⟩)

    Args:
        qubits: The qubits to measure.

    Returns:
        A handle to the measurement
        result. Access :attr:`~ket.measurement.Measurement.value` to retrieve
        the outcome as an unsigned integer, or ``None`` if not yet available.
    """
    if not isinstance(qubits, Quant):
        qubits = reduce(Quant.__add__, qubits)

    postprocessing = (
        qubits.postprocessing() if hasattr(qubits, "postprocessing") else None
    )

    return Measurement(qubits, postprocessing)


def dump(*qubits: list[Quant]) -> QuantumState:
    """Capture a full quantum state snapshot of the given qubit registers.

    Returns a :class:`~ket.quantumstate.QuantumState` object containing the
    probability amplitudes of every basis state with non-zero amplitude. This
    operation is only supported by simulators and is not available on real
    quantum hardware.


    Example:

        .. code-block:: python

            from ket import *
            p = Process()
            q = p.alloc(2)
            CNOT(H(q[0]), q[1])    # Bell state
            state = dump(q)
            print(state.states)    # {0: (0.707..+0j), 3: (0.707..+0j)}
            print(state.show())    # pretty printed state

    Args:
        *qubits: One or more qubit registers
            to capture. Multiple registers are shown as separate ket labels.

    Returns:
        A snapshot of the current
        quantum state, mapping basis state integers to their complex amplitudes.
    """

    return QuantumState(*qubits)


def sample(qubits: Quant, shots: int = 2048) -> Samples:
    """Sample the measurement outcomes of a quantum state over multiple shots.

    Runs the circuit ``shots`` times (or simulates doing so) and returns
    the empirical outcome distribution as a :class:`~ket.measurement.Samples`
    object. Unlike :func:`~ket.operations.measure`, which collapses the state
    to a single outcome, ``sample`` accumulates counts over many simulated
    shots.

    Example:

        .. code-block:: python

            from ket import *
            p = Process()
            q = p.alloc(2)
            CNOT(H(q[0]), q[1])   # Bell state
            results = sample(q, shots=4096)
            print(results.value)  # e.g., {0: 2051, 3: 2045}
            print(results.probability)  # normalized probabilities

    Args:
        qubits: The qubits to sample.
        shots: Number of measurement repetitions (shots). Defaults to
            ``2048``.

    Returns:
        A handle to the sample result,
        mapping measurement outcome integers to their counts. Returns ``None``
        if the result is not yet available (batch mode).
    """
    if not isinstance(qubits, Quant):
        qubits = reduce(Quant.__add__, qubits)

    postprocessing = (
        qubits.postprocessing() if hasattr(qubits, "postprocessing") else None
    )

    return Samples(qubits, int(shots), postprocessing)


def exp_value(hamiltonian: Hamiltonian | Pauli) -> ExpValue:
    """Calculate the expectation value of a Hamiltonian for the current quantum state.

    Registers an expectation-value computation for the given Hamiltonian or
    Pauli operator. In **live** execution mode the result is computed
    immediately; in **batch** mode it is deferred until execution.

    The Hamiltonian should be constructed using the :func:`~ket.gates.obs`
    context manager and the Pauli gate functions (:func:`~ket.gates.X`,
    :func:`~ket.gates.Y`, :func:`~ket.gates.Z`).

    Example:

        .. code-block:: python

            from ket import *
            p = Process()
            q = p.alloc(2)
            CNOT(H(q[0]), q[1])    # Prepare |Bell⟩
            with obs():
                h = X(q[0]) * X(q[1])   # XX observable
            ev = exp_value(h)
            print(ev.get())    # ⟨Bell|XX|Bell⟩ = 1.0

    Args:
        hamiltonian:
            The observable to evaluate.

    Returns:
        A handle to the expectation value result.
        Access :attr:`~ket.expv.ExpValue.value` or call ``.get()`` to retrieve
        the result as a ``float``.
    """
    return ExpValue(hamiltonian)


def using_aux(unsafe: bool = False, **names):
    """Decorator factory that automatically allocates auxiliary qubits for a gate.

    Wraps a gate function so that one or more auxiliary (ancilla) qubit
    arguments are allocated automatically by the process, rather than
    requiring the caller to manage them. The allocated qubits are passed as
    keyword arguments to the wrapped function.

    Each entry in ``names`` maps an argument name to the number of auxiliary
    qubits to allocate. The count can be either a fixed ``int`` or a
    ``Callable`` that accepts a subset of the gate's other arguments
    (by name) and returns the desired qubit count.

    When ``unsafe=True``, the gate body is executed inside a
    :meth:`~ket.base.Process.block_builder` marked as diagonal, which
    disables some safety checks. This is only appropriate for operations
    that are provably diagonal in the computational basis.

    Example:

        .. code-block:: python

            from ket import *
            @using_aux(a=lambda c: 0 if len(c) <= 2 else 1)
            def v_chain(c, t, a=None):
                # Multi-controlled X using a V-chain ancilla.
                if len(c) <= 2:
                    ctrl(c, X)(t)
                else:
                    with around(ctrl(c[:2], X), a):
                        v_chain(a + c[2:], t)
            p = Process()
            c = p.alloc(4)  # 4 control qubits
            t = p.alloc()   # 1 target qubit
            v_chain(c=c, t=t)   # ancilla allocated automatically

    Args:
        unsafe: If ``True``, mark the operation as diagonal, skipping
            certain uncomputation safety checks. Defaults to ``False``.
        **names: Keyword arguments mapping parameter names (``str``) to the
            number of auxiliary qubits needed (``int``), or a callable that
            computes the count from other gate arguments.

    Returns:
        A decorator that wraps a gate function with automatic
        auxiliary qubit allocation.
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
    """Decorator factory for self-contained, reusable quantum kernel functions.

    Creates a two-level decorator that:

    1. Accepts :class:`~ket.base.Process` constructor arguments (``*p_args``,
       ``**p_kwargs``).
    2. Returns an inner decorator that maps named parameters to qubit allocations
       and wraps a Python function as a standalone quantum kernel.

    Each call to the resulting kernel function creates a **fresh**
    :class:`~ket.base.Process`, allocates the declared qubits, calls the
    function body, and returns its return value. This makes kernels ideal for
    benchmarking, unit testing, or defining reusable quantum subroutines that
    always start from a clean state.

    Example:

        .. code-block:: python

            from math import sqrt, pi
            from ket import *
            # A kernel that measures ⟨CHSH⟩ for a Bell state.
            @kernel(num_qubits=2, simulator="dense")(a=1, b=1)
            def bell_chsh(a, b):
                CNOT(H(a), b)
                with obs():
                    a0 = Z(a)
                    a1 = X(a)
                    b0 = -(X(b) + Z(b)) / sqrt(2)
                    b1 =  (X(b) - Z(b)) / sqrt(2)
                    h = a0 * b0 + a0 * b1 + a1 * b0 - a1 * b1
                return exp_value(h).get()
            print(bell_chsh())   # Approx. 2√2 ≈ 2.828

    Args:
        *p_args: Positional arguments forwarded to the :class:`~ket.base.Process`
            constructor (e.g., ``simulator="dense"``).
        **p_kwargs: Keyword arguments forwarded to the :class:`~ket.base.Process`
            constructor (e.g., ``num_qubits=4``, ``execution="batch"``).

    Returns:
        A decorator that accepts ``**names`` (parameter-name to
        qubit-count mappings) and returns a decorator that wraps a quantum
        function as a self-contained kernel.
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
