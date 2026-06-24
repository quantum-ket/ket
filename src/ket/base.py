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

__all__ = ["Process", "Quant"]


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
    - ``"dense v2"``: Dense simulator with a smaller memory footprint.
      Not necessarily more efficient than ``"dense"``.
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
        simulator: Optional[Literal["sparse", "dense", "dense v2", "dense gpu"]] = None,
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
            super().__init__(self, ptr)
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

    def _block_qubits(self, qubits):
        for q in qubits:
            self._block_qubits[q] += 1

    def _unblock_qubits(self, qubits):
        for q in qubits:
            self._block_qubits[q] -= 1
            if self._block_qubits[q] == 0:
                del self._block_qubits[q]

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
            >>> from ket import Process
            >>> p = Process()
            >>> qubits = p.alloc(3)
            >>> print(qubits)
            <Ket 'Quant' [0, 1, 2] pid=0x...>


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
        """Allocate axillary qubits"""
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

    def param(self, *param) -> list[Parameter] | Parameter:
        """Register a parameter for gradient calculation.

        Args:
            *param: Variable-length argument list of floats.

        Returns:
            A list of :class:`~ket.base.Parameter` objects.
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
        block_str = self.gates_json()
        block = json.loads(block_str.value)
        libket["ket_string_delete"](block_str)
        return block

    def append_block(self, block, check_qubits=True):
        if check_qubits:
            proprieties_json_ptr = block.proprieties_json()
            proprieties = json.loads(proprieties_json_ptr.value)
            libket["ket_string_delete"](proprieties_json_ptr)

            if any(
                (qubit in self._aux) or (qubit in self._block_qubits)
                for qubit in proprieties["qubits_written"]
            ):
                raise RuntimeError("Fail uncomputation on axillary qubit.")

        if self._blocks:
            self._blocks[-1].append_block(block.take())
        else:
            self.__getattr__("append_block")(block.take())

    @contextmanager
    def block_builder(
        self,
        inverse=False,
        control: list[int] | None = None,
        append: bool = True,
    ):
        block = Block(self)
        self._blocks.append(block)
        try:
            yield block
        finally:
            if inverse:
                block.set(block.inverse())
            if control is not None:
                block.set(block.control(control))
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

        .. code-block:: py

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
            print(result_quant)  # <Ket 'Quant' [0, 1, 2, 3] pid=0x...>

            # Use the fist qubit to control the application of
            # a Pauli X gate on the other qubits
            ctrl(result_quant[0], X)(result_quant[1:])

            # Select qubits at specific indexes
            selected_quant = result_quant.at([0, 1])
            print(selected_quant)  # <Ket 'Quant' [0, 1] pid=0x...>

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

            .. code-block:: py

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
        """Interprets and initializes the quantum register as a quantum integer.

        Args:
            number: The initial classical integer value to set the quantum
                register to. Defaults to 0.

        Returns:
            :class:`~ket.qint.Qint`: A quantum integer data
            structure wrapping this register.
        """
        from .qint import Qint  # pylint: disable=import-outside-toplevel,cyclic-import

        return Qint(self, number)

    def as_real(self, exp: int, number: float = 0.0):
        r"""Interprets and initializes the quantum register as a fixed-point quantum real.

        The real number is represented internally as an integer scaled by :math:`2^\texttt{exp}`.
        A positive exponent provides fractional precision, while a negative exponent
        allows representing larger numbers at the cost of fine-grained precision.

        Args:
            exp: The exponent defining the fixed-point scale (value x 2**exp).
            number: The initial classical float value to set the quantum register to.
                Defaults to 0.0.

        Returns:
            :class:`~ket.qint.Qreal`: A quantum real number data structure wrapping this register.
        """
        from .qint import Qreal  # pylint: disable=import-outside-toplevel,cyclic-import

        return Qreal(self, exp, number)

    def dump_format(self):
        """Format for dump.

        Used internally em calling :func:`~ket.operations.dump`.
        """

        def dump_format(size, state):
            return f"{state:0{size}b}"

        return partial(dump_format, len(self))

    def __repr__(self):
        return f"<Ket 'Quant' {self.qubits} pid={hex(id(self.ket_process))}>"


class Parameter(HasProcess):
    """Parameter for gradient calculation.

    This class represents a parameter for gradient calculation in a quantum process. It should not
    be instanced directly, but rather obtained from the :meth:`~ket.base.Process.param`  method.
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
        """Retrieve the parameter actual value."""
        return self._param * self._multiplier

    @property
    def param(self) -> float:
        """Retrieve the original value of the parameter."""
        return self._param

    @property
    def grad(self) -> float | None:
        """Retrieve the gradient value if available."""
        if self._gradient is None:
            available, value = self.ket_process.get_gradient(self._index)
            if available.value:
                self._gradient = value.value
        return self._gradient
