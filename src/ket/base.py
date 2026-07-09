"""Base classes for quantum programming.

This module provides the base classes for quantum programming in Ket, including the
:class:`~ket.base.Process`, which is the gateway for qubit allocation and quantum execution,
and the :class:`~ket.base.Quant`, which stores the qubit's reference.

With the exception of :class:`~ket.base.Process`, the classes in this module are not
intended to be instantiated directly by the user. Instead, they are meant to be
created through provided functions.
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from contextlib import contextmanager
from ctypes import c_void_p
from functools import partial
import json
from typing import Literal, Optional, Any
import weakref

from .clib.libket import (
    Block,
    HasProcess,
    Process as LibketProcess,
    API as libket,
)

from .clib.libket.execution import LiveExecution, BatchExecution

from .clib.kbw import get_simulator

__all__ = ["Process", "Quant", "Parameter"]


class Process(LibketProcess):  # pylint: disable=too-many-instance-attributes
    """
    Quantum program process.

    A :class:`~ket.base.Process` in Ket is responsible for preparing and executing quantum circuits.
    It serves as a direct interface to the underlying Rust runtime library. The primary way to
    interact with a process is through the :meth:`~ket.base.Process.alloc` method to allocate
    qubits.

    Example:

        .. code-block:: python

            from ket import Process
            p = Process()
            qubits = p.alloc(10)  # Allocate 10 qubits

    By default, quantum execution is handled by the KBW simulator in **sparse** mode with support
    for up to 32 qubits. In sparse mode, qubits are represented using a data structure similar to a
    sparse matrix. This mode performs well when the quantum state involves the superposition of a
    small number of basis states, such as GHZ states, and is suitable as a general default when the
    number of qubits is unknown.

    The **dense** simulation mode, on the other hand, has exponential time complexity in the number
    of qubits. It leverages CPU parallelism more effectively, but requires careful management of the
    number of qubits, defaulting to 12. The choice between sparse and dense simulation depends on
    the specific quantum algorithm being implemented, as each mode has trade-offs in performance
    and scalability.

    The execution mode of the simulator can be set to either ``"live"`` or ``"batch"``:

    - **Live** (default): Quantum instructions are executed immediately upon invocation. This mode
      is ideal for interactive simulation.

    - **Batch**: Quantum instructions are queued and executed only at the a measurement result is
      requested. This mode better reflects the behavior of real quantum hardware and is recommended
      for preparing code for deployment to QPUs.

    Batch Execution Example:

    .. code-block:: python

        from ket import *
        p = Process(execution="batch")
        a, b = p.alloc(2)
        CNOT(H(a), b)  # Prepare a Bell state
        d = sample(a + b)
        print(d.get())  # Execution happens here
        CNOT(a, b)  # Raises an error: process already executed

    Live Execution Example:

    .. code-block:: python

        from ket import *
        p = Process(execution="live")
        a, b = p.alloc(2)
        CNOT(H(a), b)  # Prepare a Bell state
        print(sample(a + b).get())  # Output is available immediately
        CNOT(a, b)
        H(a)
        print(sample(a + b).get())


    :Simulators:

    KBW provides four simulators with different performance characteristics.
    The best simulator to use depends on the number of qubits being simulated
    and on the specific quantum algorithm. Benchmarking is recommended to
    determine the most suitable simulator for a given workload.

    - ``"sparse"``: Sparse simulator with limited multithreading capabilities.
    - ``"dense"``: Dense simulator with good multithreaded performance.
    - ``"dense gpu"``: Dense simulator designed to run on most GPUs, including
      integrated Intel, AMD, and Apple GPUs, as well as NVIDIA GPUs. This simulator
      is generally recommended for simulations involving a large number of qubits.


    Args:
        execution_target: Quantum execution target object. If not provided, the KBW simulator
            is used.
        num_qubits: Number of qubits for the KBW simulator.
            Defaults to 32 for sparse mode, or 12 for dense mode.
        simulator: Simulation mode for the KBW simulator. Defaults to ``"sparse"``.
        execution: Execution mode for the KBW simulator, either ``"live"`` or ``"batch"``.
            Defaults to ``"live"``.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        execution_target: BatchExecution | LiveExecution | None = None,
        num_qubits: Optional[int] = None,
        simulator: Optional[Literal["sparse", "dense", "dense gpu"]] = None,
        execution: Optional[Literal["live", "batch"]] = None,
        gradient: bool = False,
        **kwargs,
    ):

        if execution_target is not None and any(
            map(lambda a: a is not None, [num_qubits, simulator, execution])
        ):
            raise ValueError("Cannot specify arguments if configuration is provided")

        if execution_target is not None:
            self.configuration = execution_target
            ptr = self.configuration.connect()
            assert isinstance(ptr, c_void_p)
            super().__init__(ptr)
        else:
            simulator = "sparse" if simulator is None else simulator
            num_qubits = (
                (32 if simulator == "sparse" else 12)
                if num_qubits is None
                else num_qubits
            )
            super().__init__(
                get_simulator(
                    num_qubits=num_qubits,
                    simulator=simulator,
                    execution="live" if execution is None else execution,
                    gradient=gradient,
                    **kwargs,
                ),
            )

        self._exp_value_count = 0
        self._exp_values = None

        self._qubit_pool = []
        self._aux = set()
        self._blocked_qubits = defaultdict(int)

        self._blocks = []
        self._is_diagonal = 0
        self._is_permutation = 0

    def _block_qubits(self, qubits):
        for q in qubits:
            self._blocked_qubits[q] += 1

    def _unblock_qubits(self, qubits):
        for q in qubits:
            self._blocked_qubits[q] -= 1
            if self._blocked_qubits[q] == 0:
                del self._blocked_qubits[q]

    def _get_exp_value_index(self) -> int:
        index = self._exp_value_count
        self._exp_value_count += 1
        return index

    def _get_exp_value(self, index, execute=False):
        if execute:
            self.execute()
            result_ptr = self.read_exp_value()
            self._exp_values = json.loads(result_ptr.value.decode("utf-8"))
            libket["ket_string_delete"](result_ptr)

        if self._exp_values is None:
            return None

        return self._exp_values[index]

    def _alloc(self) -> int:
        if self._qubit_pool:
            return self._qubit_pool.pop(0)
        return super().__getattr__("alloc")().value

    def _is_aux(self, index):
        return index in self._aux

    def alloc(self, num_qubits: int = 1) -> Quant:
        """Allocate qubits and return a :class:`~ket.base.Quant` object.

        Each qubit is assigned a unique index, and the
        resulting :class:`~ket.base.Quant` object encapsulates the allocated qubits along with a
        reference to the parent :class:`~ket.base.Process` object.

        Example:
            .. code-block:: python

                from ket import Process
                p = Process()
                qubits = p.alloc(3)
                print(qubits)
                # <Ket 'Quant' [0, 1, 2] pid=0x...>


        Args:
            num_qubits: The number of qubits to allocate. Defaults to 1.

        Returns:
            A list like object representing the allocated qubits.
        """

        if num_qubits < 1:
            raise ValueError("Cannot allocate less than 1 qubit")

        qubits_index = [self._alloc() for _ in range(num_qubits)]
        return Quant(qubits=qubits_index, process=self)

    def alloc_aux(self, num_qubits: int = 1) -> Quant:
        """Allocate auxiliary qubits managed by the process for uncomputation.

        Auxiliary (ancilla) qubits are temporary qubits used in intermediate
        computation steps, such as in multi-controlled gates or temporary
        registers. The process tracks them separately and prevents accidental
        measurement. When the returned :class:`~ket.base.Quant` object goes out
        of scope or is used as a context manager, the auxiliary qubits are
        automatically freed and returned to the internal qubit pool for reuse.

        The recommended usage is as a context manager with the ``with`` statement:

        Example:
            .. code-block:: python

                from ket import Process, ctrl, X
                p = Process()
                c, t = p.alloc(), p.alloc()
                with p.alloc_aux() as aux:         # aux is freed on exit
                    with around(ctrl(c, X), aux):  # use aux as intermediate
                        ctrl(aux, X)(t)            # fanout to target

        Args:
            num_qubits: The number of auxiliary qubits to allocate.
                Defaults to 1.

        Returns:
            A :class:`~ket.base.Quant` object containing
            the auxiliary qubits. It supports use as a context manager: on exit
            the qubits are returned to the qubit pool.

        Raises:
            ValueError: If ``num_qubits`` is less than 1.
        """
        if num_qubits < 1:
            raise ValueError("Cannot allocate less than 1 qubit")

        aux_index = [self._alloc() for _ in range(num_qubits)]

        self._aux.update(aux_index)

        return Quant(
            qubits=aux_index,
            process=self,
            undo=lambda: self._free_aux(aux_index),
        )

    def _free_aux(self, qubits: list[int]):
        if any(q not in self._aux for q in qubits):
            raise RuntimeError("Cannot free an non-auxiliary qubit")

        self._aux.difference_update(qubits)
        self._qubit_pool.extend(qubits)

    def param(self, *param: float) -> list[Parameter] | Parameter:
        """Register one or more differentiable parameters for gradient computation.

        Each numeric value is wrapped in a :class:`~ket.base.Parameter` object
        that can be scaled (via multiplication or division) and passed directly
        to parameterized gates such as :func:`~ket.gates.RX`, :func:`~ket.gates.RY`,
        or :func:`~ket.gates.P`. After circuit execution, the gradient of the
        expected value with respect to each parameter can be retrieved via
        :attr:`~ket.base.Parameter.grad`.

        .. note::
            Gradient computation requires creating the process with
            ``gradient=True``.

        Example:
            .. code-block:: python

                from math import pi
                from ket import Process, RX, exp_value
                from ket.gates import obs
                import ket
                p = Process(gradient=True)
                theta = p.param(pi / 4)
                q = p.alloc()
                RX(theta, q)
                with obs():
                    h = ket.Z(q)
                ev = ket.exp_value(h)
                ev.get()        # trigger execution
                theta.grad      # d<Z>/d(theta)

        Args:
            *param: One or more initial float values for the parameters.

        Returns:
            A single :class:`~ket.base.Parameter` if one value is provided,
            otherwise a list of :class:`~ket.base.Parameter` objects.
        """
        parameters = [
            Parameter(process=self, index=self.set_parameter(p).value, value=p)
            for p in param
        ]
        if len(param) == 1:
            return parameters[0]
        return parameters

    def __repr__(self) -> str:
        return f"<Ket 'Process' id={hex(id(self))}>"

    def gates(self):
        """Return the gate sequence of the process as a parsed JSON object.

        Serializes the internal circuit representation into a JSON-compatible
        Python object (typically a list of gate dictionaries). This is primarily
        used for inspection, debugging, or exporting the circuit.

        Returns:
            A JSON-parsed object representing the gate sequence of the
            current process.

        Example:
            .. code-block:: python

                from ket import Process, H, X
                p = Process()
                q = p.alloc(2)
                H(q[0])
                X(q[1])
                circuit = p.gates()
                print(type(circuit))
                # <class 'list'>
        """
        block_str = self.gates_json()
        block = json.loads(block_str.value)
        libket["ket_string_delete"](block_str)
        return block

    def append_block(self, block, check_qubits=True):
        """Append a pre-built gate block to the process circuit.

        .. caution::
            This is an internal method and is not intended for direct use by
            library consumers. Use quantum gate functions instead.

        If there is a currently active nested block (e.g., inside a
        :meth:`~ket.base.Process.block_builder` context), the block is appended
        to the innermost active block. Otherwise, it is appended directly to the
        process circuit.

        When ``check_qubits`` is ``True`` (the default), the method validates that
        no uncomputation rules are violated, for example, it prevents applying
        non-diagonal operations to auxiliary qubits that are currently blocked.


        Args:
            block: The compiled gate block to append.
            check_qubits: If ``True``, validate qubit operation rules
                before appending. Defaults to ``True``.

        Raises:
            RuntimeError: If ``check_qubits`` is ``True`` and the block contains
                an operation that violates uncomputation constraints.
        """
        if self._blocks:
            self._blocks[-1].append_block(block.take())
        else:
            if check_qubits:
                proprieties_json_ptr = block.proprieties_json()
                proprieties = json.loads(proprieties_json_ptr.value)
                libket["ket_string_delete"](proprieties_json_ptr)

                for target, op in proprieties.items():
                    target = int(target)
                    if (op["propriety"] in ["Permutation", "Unitary"]) and (
                        (target in self._blocked_qubits) or self._is_aux(target)
                    ):
                        raise RuntimeError("Operation violates uncomputation.")
            super().__getattr__("append_block")(block.take())

    @contextmanager
    def block_builder(
        self,
        inverse=False,
        control: list[int] | None = None,
        append: bool = True,
        diagonal=False,
        permutation=False,
    ):
        """Context manager for constructing and optionally appending a gate block.

        .. caution::
            This is an internal method. Prefer using the high-level gate API and
            context managers (:func:`~ket.operations.control`,
            :func:`~ket.operations.inverse`, :func:`~ket.operations.around`).

        All quantum operations performed inside the ``with`` block are collected
        into a single circuit ``Block`` object. On exit, the block can optionally
        be inverted, wrapped in a controlled operation, and/or appended to the
        process circuit.

        This is the primary building primitive used internally by gate functions
        and higher-level operations like :func:`~ket.operations.around` and
        :func:`~ket.operations.inverse`.


        Args:
            inverse: If ``True``, invert the block before appending.
                Defaults to ``False``.
            control: If provided, wrap the block in a
                multi-qubit controlled operation on the specified qubit indices.
                Defaults to ``None``.
            append: If ``True``, append the block to the process circuit
                on exit. Set to ``False`` to obtain the block without appending
                it. Defaults to ``True``.
            diagonal: If ``True``, mark the block as a diagonal operation.
                Defaults to ``False``.
            permutation: If ``True``, mark the block as a permutation
                operation. Defaults to ``False``.

        Yields:
            Block: The in-progress gate block being constructed.
        """
        block = Block(self)
        self._blocks.append(block)
        if diagonal:
            self._is_diagonal += 1
        elif permutation:
            self._is_permutation += 1

        try:
            yield block
        finally:
            if inverse:
                block.set(block.inverse())
            if control is not None:
                block.set(block.control(control))

            if self._is_diagonal > 0:
                block.set_as_diagonal()
            elif self._is_permutation > 0:
                block.set_as_permutation()

            if diagonal:
                self._is_diagonal -= 1
            elif permutation:
                self._is_permutation -= 1

            self._blocks.pop()
            if append:
                self.append_block(block)


class Quant(HasProcess):
    """List of qubits.

    This class represents a list of qubit indices within a quantum process. Direct instantiation
    of this class is not recommended. Instead, it should be created by calling the
    :meth:`~ket.base.Process.alloc` method.

    A :class:`~ket.base.Quant` serves as a fundamental quantum object where quantum operations
    should be applied.

    Example:

        .. code-block:: python

            from ket import *
            # Create a quantum process
            p = Process()
            # Allocate 2 qubits
            q1 = p.alloc(2)
            # Apply a Hadamard gates on the first qubit of `q1`
            H(q1[0])
            # Allocate more 2 qubits
            q2 = p.alloc(2)
            # Concatenate two Quant objects
            result_quant = q1 + q2
            print(result_quant)
            # <Ket 'Quant' [0, 1, 2, 3] pid=0x...>
            # Use the fist qubit to control the application of
            # a Pauli X gate on the other qubits
            ctrl(result_quant[0], X)(result_quant[1:])
            # Select qubits at specific indexes
            selected_quant = result_quant.at([0, 1])
            print(selected_quant)
            # <Ket 'Quant' [0, 1] pid=0x...>

    Supported operations:

    - Addition (``+``): Concatenates two :class:`~ket.base.Quant` objects.
      The processes must be the same.
    - Indexing (``[index]``): Returns a new :class:`~ket.base.Quant` object with selected qubits
      based on the provided index.
    - Iteration (``for q in qubits``): Allows iterating over qubits in a :class:`~ket.base.Quant`
      object.
    - Reversal (``reversed(qubits)``): Returns a new :class:`~ket.base.Quant` object with reversed
      qubits.
    - Length (``len(qubits)``): Returns the number of qubits in the :class:`~ket.base.Quant` object.

    """

    def __init__(self, *, qubits: list[int], process: Process, undo=None, source=None):
        super().__init__(ket_process=process)

        self.qubits = qubits
        self._finalizer = weakref.finalize(self, undo) if undo is not None else None
        self.source = source

    def __add__(self, other: Quant) -> Quant:
        """Concatenate two :class:`~ket.base.Quant` objects into a single register.

        Creates a new :class:`~ket.base.Quant` whose qubit list is the ordered
        concatenation of ``self`` followed by ``other``. Both objects must belong
        to the same :class:`~ket.base.Process` and must not share any qubit indices.

        Example:
            .. code-block:: python

                from ket import Process
                p = Process()
                q1 = p.alloc(2)
                q2 = p.alloc(2)
                combined = q1 + q2
                print(len(combined))
                # 4

        Args:
            other: The qubit register to append.

        Returns:
            A new register containing all qubits from
            ``self`` followed by all qubits from ``other``.

        Raises:
            ValueError: If ``other`` belongs to a different process, or if the
                two registers share any qubit indices.
        """
        if not isinstance(other, Quant):
            return NotImplemented
        if self.ket_process is not other.ket_process:
            raise ValueError("Cannot concatenate qubits from different processes")
        if any(qubit in other.qubits for qubit in self.qubits):
            raise ValueError("Cannot concatenate qubits with overlapping indices")
        return Quant(
            qubits=self.qubits + other.qubits,
            process=self.ket_process,
            source=[self, other],
        )

    def at(self, index: list[int]) -> Quant:
        """Return a subset of qubits at specified indices.

        Create a new :class:`~ket.base.Quant` object with qubit references at the positions defined
        by the provided ``index`` list.

        Example:

            .. code-block:: python

                from ket import *
                # Create a quantum process
                p = Process()
                # Allocate 5 qubits
                q = p.alloc(5)
                # Select qubits at odd indices (1, 3)
                odd_qubits = q.at([1, 3])

        Args:
            index: List of indices specifying the positions of qubits to be included in the
                new :class:`~ket.base.Quant`.

        Returns:
            A new :class:`~ket.base.Quant` object containing the selected qubits.
        """

        return Quant(
            qubits=[self.qubits[i] for i in index],
            process=self.ket_process,
            source=self,
        )

    def __reversed__(self):
        return Quant(
            qubits=list(reversed(self.qubits)),
            process=self.ket_process,
            source=self,
        )

    def __getitem__(self, key):
        qubits = self.qubits.__getitem__(key)
        return Quant(
            qubits=qubits if isinstance(qubits, list) else [qubits],
            process=self.ket_process,
            source=self,
        )

    def __len__(self):
        return len(self.qubits)

    def __eq__(self, other: Any):
        if not isinstance(other, int):
            return NotImplemented

        from .operations import (  # pylint: disable=import-outside-toplevel,cyclic-import
            undo,
            _flip_to_control,
        )

        return undo(_flip_to_control(other), self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._finalizer is not None and self._finalizer.alive:
            self._finalizer()

    def as_int(self, number: int = 0):
        """Interpret and initialize this quantum register as a quantum integer.

        Wraps the register as a :class:`~ket.qint.Qint`, enabling quantum
        arithmetic operations (addition, subtraction, comparison, etc.) on the
        underlying qubits. The register is initialized to the given classical
        integer value using :func:`~ket.gates.X` gates.

        The :class:`~ket.qint.Qint` uses a two's-complement signed representation
        internally.

        Example:
            .. code-block:: python

                from ket import Process, measure
                p = Process()
                q = p.alloc(5)
                qi = q.as_int(5)    # register initialized to |5⟩
                qi += 3             # in-place addition: |5⟩ → |8⟩
                print(measure(qi).value)
                # 8

        Args:
            number: The initial classical integer value to encode into the
                quantum register. Defaults to ``0``.

        Returns:
            A quantum integer wrapping this register,
            initialized to ``number``.
        """
        from .qint import Qint  # pylint: disable=import-outside-toplevel,cyclic-import

        return Qint(self, number)

    def as_real(self, exp: int, number: float = 0.0):
        r"""Interpret and initialize this quantum register as a fixed-point quantum real number.

        Wraps the register as a :class:`~ket.qint.Qreal`, enabling quantum
        arithmetic operations on floating-point values encoded in a fixed-point
        binary representation.

        The real number is stored internally as an integer scaled by
        :math:`2^{\texttt{exp}}`:

        - A **positive** ``exp`` increases fractional precision (smaller representable
          step size of :math:`2^{-\texttt{exp}}`).
        - A **negative** ``exp`` increases the representable magnitude at the cost of
          precision.

        Example:
            .. code-block:: python

                from ket import Process, measure
                p = Process()
                q = p.alloc(8)         # 8 qubits for fixed-point
                qr = q.as_real(4, 1.5) # precision: 1/16, initialized to 1.5
                qr += 0.25             # in-place addition
                print(measure(qr).value)
                # 1.75

        Args:
            exp: The exponent defining the fixed-point scale. The stored
                integer ``n`` represents the real value ``n / 2**exp``.
            number: The initial classical float value to encode into the
                quantum register. Defaults to ``0.0``.

        Returns:
            A quantum real number wrapping this register,
            initialized to ``number``.
        """
        from .qint import Qreal  # pylint: disable=import-outside-toplevel,cyclic-import

        return Qreal(self, exp, number)

    def dump_format(self):
        """Return the state-formatting callable used by :func:`~ket.operations.dump`.

        Provides a function that converts a raw integer basis-state index into a
        zero-padded binary string of the correct width for this register. This is
        used internally by :class:`~ket.quantumstate.QuantumState` to display
        multi-register states with per-register labels.

        Returns:
            A function that accepts an integer basis-state
            value and returns its binary string representation (zero-padded to
            ``len(self)`` bits).
        """

        def dump_format(size, state):
            return f"{state:0{size}b}"

        return partial(dump_format, len(self))

    def __repr__(self):
        return f"<Ket 'Quant' {self.qubits} pid={hex(id(self.ket_process))}>"


class Parameter(HasProcess):
    """A differentiable scalar parameter for variational quantum circuits.

    This class wraps a floating-point value and registers it with a
    :class:`~ket.base.Process` so that the runtime can compute analytical
    gradients (e.g., using the parameter-shift rule). It supports scalar
    arithmetic (``*``, ``/``, unary ``-``) so that a single registered
    parameter can be reused across multiple gates with different multipliers.

    .. important::
        Do not instantiate this class directly. Obtain :class:`~ket.base.Parameter`
        objects from :meth:`~ket.base.Process.param`.

    Example:
        .. code-block:: python

            from math import pi
            from ket import Process, RX, RZ
            p = Process(gradient=True)
            theta = p.param(pi / 3)
            q = p.alloc()
            RX(theta, q)       # use the parameter directly
            RZ(theta / 2, q)   # reuse with a scaled copy
    """

    def __init__(self, process, index, value, multiplier=1):
        super().__init__(ket_process=process)
        self._index = index
        self._param = value
        self._multiplier = multiplier
        self._gradient = None

    def __mul__(self, other: float) -> Parameter:
        other = float(other)
        return Parameter(
            self.ket_process,
            self._index,
            self._param,
            self._multiplier * other,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: float) -> Parameter:
        other = float(other)
        return Parameter(
            self.ket_process,
            self._index,
            self._param,
            self._multiplier / other,
        )

    def __neg__(self) -> Parameter:
        return Parameter(
            self.ket_process,
            self._index,
            self._param,
            -self._multiplier,
        )

    def __repr__(self):
        return (
            f"<Ket 'Parameter' param={self._param}, value={self.value},"
            + f" index={self._index}, pid={hex(id(self.ket_process))}>"
        )

    def __float__(self):
        return self.value

    @property
    def value(self) -> float:
        """The effective (scaled) value of this parameter.

        Returns the product of the original registered value and any
        multiplier applied via arithmetic operators.

        Returns:
            ``param * multiplier``.
        """
        return self._param * self._multiplier

    @property
    def param(self) -> float:
        """The original, unscaled value as registered with the process.

        Returns:
            The raw value passed when creating this parameter via
            :meth:`~ket.base.Process.param`.
        """
        return self._param

    @property
    def grad(self) -> float | None:
        """The gradient of the circuit output with respect to this parameter.

        Lazily fetched from the process after circuit execution. Returns
        ``None`` if the gradient has not yet been computed (i.e., the process
        has not executed) or if gradient computation was not enabled on the
        :class:`~ket.base.Process`.

        Returns:
            The gradient value, or ``None`` if unavailable.

        Example:
            .. code-block:: python

                from math import pi
                import ket
                p = ket.Process(gradient=True)
                theta = p.param(pi / 4)
                q = p.alloc()
                ket.RX(theta, q)
                with ket.gates.obs():
                    h = ket.Z(q)
                ev = ket.exp_value(h)
                ev.get()         # execute and retrieve expected value
                print(theta.grad)  # d<Z>/d(theta)
        """
        if self._gradient is None:
            available, value = self.ket_process.get_gradient(self._index)
            if available.value:
                self._gradient = value.value
        return self._gradient
