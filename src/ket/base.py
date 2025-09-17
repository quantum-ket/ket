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


from ctypes import c_size_t, c_uint8, c_void_p
from json import loads
from typing import Literal, Optional, Any

from .clib.libket import LiveExecution, Process as LibketProcess, BatchExecution
from .clib.kbw import get_simulator

try:
    import plotly.graph_objs as go
    import plotly.express as px

    VISUALIZE = True
except ImportError:
    VISUALIZE = False

__all__ = [
    "Process",
    "Quant",
    "Measurement",
    "Samples",
]


class Process(LibketProcess):
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

    Args:
        execution_target: Quantum execution target object. If not provided, the KBW simulator
            is used.
        num_qubits: Number of qubits for the KBW simulator.
            Defaults to 32 for sparse mode, or 12 for dense mode.
        simulator: Simulation mode for the KBW simulator, either ``"sparse"`` or ``"dense"``.
            Defaults to ``"sparse"``.
        execution: Execution mode for the KBW simulator, either ``"live"`` or ``"batch"``.
            Defaults to ``"live"``.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        execution_target: BatchExecution | LiveExecution | None = None,
        num_qubits: Optional[int] = None,
        simulator: Optional[Literal["sparse", "dense", "dense v2"]] = None,
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
                )
            )

        self._metadata_buffer_size = 512
        self._metadata_buffer = (c_uint8 * self._metadata_buffer_size)()
        self._buffer_size = 2048
        self._buffer = (c_uint8 * self._buffer_size)()

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

        qubits_index = [self.allocate_qubit().value for _ in range(num_qubits)]
        return Quant(qubits=qubits_index, process=self)

    def _get_ket_process(self):
        return self

    def get_instructions(self) -> list[dict]:
        """Retrieve quantum instructions from the process.

        The format of the instructions is defined in the runtime library Libket.

        Example:
            >>> from ket import *
            >>> p = Process()
            >>> a, b = p.alloc(2)
            >>> CNOT(H(a), b)
            >>> # Get quantum instructions
            >>> instructions = p.get_instructions()
            >>> pprint(instructions)
            [{'Gate': {'control': [],
                       'gate': 'Hadamard',
                       'target': {'Main': {'index': 0}}}},
             {'Gate': {'control': [{'Main': {'index': 0}}],
                       'gate': 'PauliX',
                       'target': {'Main': {'index': 1}}}}]


        Returns:
            A list of dictionaries containing quantum instructions extracted from the process.
        """
        write_size = self.instructions_json(self._buffer, self._buffer_size)
        if write_size.value > self._buffer_size:
            self._buffer_size = write_size.value + 1
            self._buffer = (c_uint8 * self._buffer_size)()
            return self.get_instructions()

        return loads(bytearray(self._buffer[: write_size.value]))

    def get_mapped_instructions(self) -> list[dict] | None:
        """Retrieve quantum instructions after the circuit mapping.

        The format of the instructions is defined in the runtime library Libket.

        The instructions are extracted after the circuit mapping step, which is
        performed during the compilation process. A qubit coupling graph must be passed to the
        process for the quantum circuit mapping to happen. Note that at this point, the single
        qubit gates have not been translated to the native gate set of the quantum hardware yet.

        Example:

            >>> n = 4
            >>> coupling_graph = [(0, 1), (1, 2), (2, 3), (3, 0)]
            >>> p = Process(num_qubits=n, coupling_graph=coupling_graph)
            >>> q = p.alloc(n)
            >>> ctrl(H(q[0]), X)(q[1:])
            >>> m = measure(q)
            >>> p.prepare_for_execution()
            >>> pprint(p.get_mapped_instructions())
            [{'Gate': {'control': [], 'gate': 'Hadamard', 'target': {'index': 0}}},
             {'Gate': {'control': [{'index': 0}],
                       'gate': 'PauliX',
                       'target': {'index': 1}}},
             'Identity',
             {'Gate': {'control': [{'index': 3}],
                       'gate': 'PauliX',
                       'target': {'index': 0}}},
             {'Gate': {'control': [{'index': 0}],
                       'gate': 'PauliX',
                       'target': {'index': 3}}},
             {'Gate': {'control': [{'index': 3}],
                       'gate': 'PauliX',
                       'target': {'index': 2}}},
             {'Measure': {'index': 0,
                          'qubits': [{'index': 3},
                                     {'index': 1},
                                     {'index': 0},
                                     {'index': 2}]}}]


        Returns:
            A list of dictionaries containing quantum instructions extracted from the process
            if the process has been transpiled, otherwise None.

        """
        write_size = self.isa_instructions_json(self._buffer, self._buffer_size)
        if write_size.value > self._buffer_size:
            self._buffer_size = write_size.value + 1
            self._buffer = (c_uint8 * self._buffer_size)()
            return self.get_isa_instructions()

        return loads(bytearray(self._buffer[: write_size.value]))

    def get_metadata(self) -> dict[str, Any]:
        """Retrieve metadata from the quantum process.

        Example:

            >>> n = 4
            >>> coupling_graph = [(0, 1), (1, 2), (2, 3), (3, 0)]
            >>> p = Process(num_qubits=n, coupling_graph=coupling_graph)
            >>> q = p.alloc(n)
            >>> ctrl(H(q[0]), X)(q[1:])
            >>> m = measure(q)
            >>> p.prepare_for_execution()
            >>> pprint(p.get_mapped_instructions())
            {'allocated_qubits': 4,
             'decomposition': {'NoAuxCX': 3},
             'logical_circuit_depth': 3,
             'logical_gate_count': {'1': 1, '2': 3},
             'physical_circuit_depth': 5,
             'physical_gate_count': {'1': 1, '2': 4},
             'terminated': True}


        Returns:
            A dictionary containing metadata information extracted from the process.

        """

        write_size = self.metadata_json(
            self._metadata_buffer, self._metadata_buffer_size
        )
        if write_size.value > self._metadata_buffer_size:
            self._metadata_buffer_size = write_size.value + 1
            self._metadata_buffer = (c_uint8 * self._metadata_buffer_size)()
            return self.get_metadata()

        return loads(bytearray(self._metadata_buffer[: write_size.value]))

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

    def save(self):
        """
        Save the quantum simulation state.

        This method saves the current state of the quantum simulation. It works only
        in simulated execution, provided that the simulator supports state saving.
        If state saving is not available, no error will be raised.

        The structure of the saved state depends on the simulator and is not interchangeable
        between different simulators. Note that the saved state does not retain information
        about the order of qubit allocation. Using the saved state may result in unintended
        behavior if the qubit allocation order differs when the state is loaded.
        """

        write_size = self.save_sim_state(self._buffer, self._buffer_size)
        if write_size.value > self._buffer_size:
            self._buffer_size = write_size.value + 1
            self._buffer = (c_uint8 * self._buffer_size)()
            return self.save()

        return bytearray(self._buffer[: write_size.value])

    def load(self, data):
        """
        Loads the given data into the simulation state.

        The data must be generated by a call to the method `save` with the same simulator.
        If the state load fails in the simulator, the state is not altered,
        and it may raise no error.

        Args:
            data: Quantum state. The data format depends on the simulator.
        """

        data_len = len(data)
        data = (c_uint8 * data_len)(*data)
        self.load_sim_state(data, data_len)

    def __repr__(self) -> str:
        return f"<Ket 'Process' id={hex(id(self))}>"


class Quant:
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

    def __init__(self, *, qubits: list[int], process: Process):
        self.qubits = qubits
        self.process = process

    def _get_ket_process(self):
        return self.process

    def __add__(self, other: Quant) -> Quant:
        if self.process is not other.process:
            raise ValueError("Cannot concatenate qubits from different processes")
        if any(qubit in other.qubits for qubit in self.qubits):
            raise ValueError("Cannot concatenate qubits with overlapping indices")
        return Quant(qubits=self.qubits + other.qubits, process=self.process)

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

        return Quant(qubits=[self.qubits[i] for i in index], process=self.process)

    def __reversed__(self):
        return Quant(qubits=list(reversed(self.qubits)), process=self.process)

    def __getitem__(self, key):
        qubits = self.qubits.__getitem__(key)
        return Quant(
            qubits=qubits if isinstance(qubits, list) else [qubits],
            process=self.process,
        )

    class _QuantIter:
        def __init__(self, q: Quant):
            self.q = q
            self.idx = -1
            self.size = len(q.qubits)

        def __next__(self):
            self.idx += 1
            if self.idx < self.size:
                return self.q[self.idx]
            raise StopIteration

        def __iter__(self):
            return self

    def __iter__(self):
        return self._QuantIter(self)

    def __len__(self):
        return len(self.qubits)

    def __repr__(self):
        return f"<Ket 'Quant' {self.qubits} pid={hex(id(self.process))}>"


class Measurement:
    """Quantum measurement result.

    This class holds a reference for a measurement result. The result may not be available right
    after the measurement call, especially in batch execution.

    To read the value, access the attribute :attr:`~ket.base.Measurement.value`. If the value is not
    available, the measurement will return `None`; otherwise, it will return an unsigned integer.

    You can instantiate this class by calling the :func:`~ket.operations.measure` function.

    Example:

        .. code-block:: python

            from ket import *

            p = Process()
            q = p.alloc(2)
            CNOT(H(q[0]), q[1])
            result = measure(q)
            print(result.value)  # 0 or 3
    """

    def __init__(self, qubits: Quant):
        self.process = qubits.process
        self.qubits = [qubits[i : i + 64] for i in range(0, len(qubits), 64)]
        self.indexes = [
            self.process.measure(
                (c_size_t * len(qubit.qubits))(*qubit.qubits), len(qubit.qubits)
            ).value
            for qubit in self.qubits
        ]
        self._value = None

    def _get_ket_process(self):
        return self.process

    def _check(self):
        if self._value is None:
            available, values = zip(
                *(self.process.get_measurement(index) for index in self.indexes)
            )
            if all(map(lambda a: a.value, available)):
                self._value = 0
                for value, qubit in zip(values, self.qubits):
                    self._value <<= len(qubit)
                    self._value |= value.value

    @property
    def value(self) -> int | None:
        """Retrieve the measurement value if available."""
        self._check()
        return self._value

    def get(self) -> int:
        """Retrieve the measurement value.

        If the value is not available, the quantum process will execute to get the result.
        """

        self._check()
        if self._value is None:
            self.process.execute()
        return self.value

    def __repr__(self):
        return (
            f"<Ket 'Measurement' indexes={self.indexes}, "
            f"value={self.value}, pid={hex(id(self.process))}>"
        )


class Samples:
    """Quantum state measurement samples.

    This class holds a reference for a measurement sample result. The result may not be available
    right after the sample call, especially in batch execution.

    To read the value, access the attribute :attr:`~ket.base.Sample.value`. If the value is not
    available, the measurement will return `None`; otherwise, it will return a dictionary mapping
    measurement outcomes to their respective counts.

    You can instantiate this class by calling the :func:`~ket.operations.sample` function.

    Example:

        .. code-block:: py

            from ket import *

            p = Process()
            q = p.alloc(2)
            CNOT(H(q[0]), q[1])
            results = sample(q)

            print(results.value)
            # {0: 1042, 3: 1006}

    Args:
        qubits: Qubits for which the measurement samples are obtained.
        shots: Number of measurement shots (default is 2048).

    """

    def __init__(self, qubits: Quant, shots: int = 2048):
        self.qubits = qubits
        self.process = qubits.process
        self.index = self.process.sample(
            (c_size_t * len(qubits.qubits))(*qubits.qubits), len(qubits.qubits), shots
        ).value
        self._value = None
        self.shots = shots

    def _check(self):
        if self._value is None:
            (
                available,
                states,
                count,
                size,
            ) = self.process.get_sample(self.index)
            if available.value:
                self._value = dict(zip(states[: size.value], count[: size.value]))

    @property
    def value(self) -> dict[int, int] | None:
        """Retrieve the measurement samples if available."""
        self._check()
        return self._value

    def get(self) -> dict[int, int]:
        """Retrieve the measurement samples.

        If the value is not available, the quantum process will execute to get the result.
        """

        self._check()
        if self._value is None:
            self.process.execute()
        return self.value

    def histogram(self, mode: Literal["bin", "dec"] = "dec", **kwargs) -> go.Figure:
        """Generate a histogram representing the sample.

        This method creates a histogram visualizing the probability distribution
        of the sample.

        Note:
            This method requires additional dependencies from ``ket-lang[plot]``.

            Install with: ``pip install ket-lang[plot]``.

        Args:
            mode: If ``"bin"``, display the states in binary format. If ``"dec"``,
                display the states in decimal format. Defaults to ``"dec"``.
            **kwargs: Additional keyword arguments passed to :func:`plotly.express.bar`.

        Returns:
            Histogram of sample measurement.
        """
        _check_visualize()

        state = list(self.get().keys())
        state_text = (
            list(map(lambda s: f"|{s:0{len(self.qubits)}b}⟩", state))
            if mode == "bin"
            else state
        )

        data = {
            "State": state,
            "Count": list(self.get().values()),
        }

        fig = px.bar(
            data,
            x="State",
            y="Count",
            **kwargs,
        )

        fig.update_layout(
            xaxis={
                "tickmode": "array",
                "tickvals": state,
                "ticktext": state_text,
            },
            bargap=0.75,
        )

        return fig

    def __repr__(self) -> str:
        return f"<Ket 'Samples' index={self.index}, pid={hex(id(self.process))}>"


def _check_visualize():
    if not VISUALIZE:
        raise RuntimeError(
            "Visualization optional dependence are required. Install with: "
            "pip install ket-lang[plot]"
        )


class Parameter:
    """Parameter for gradient calculation.

    This class represents a parameter for gradient calculation in a quantum process. It should not
    be instanced directly, but rather obtained from the :meth:`~ket.base.Process.param`  method.
    """

    def __init__(self, process, index, value, multiplier=1):
        self._process = process
        self._index = index
        self._param = value
        self._multiplier = multiplier
        self._gradient = None

    def __mul__(self, other: float) -> Parameter:
        other = float(other)
        return Parameter(
            self._process,
            self._index,
            self._param,
            self._multiplier * other,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: float) -> Parameter:
        other = float(other)
        return Parameter(
            self._process,
            self._index,
            self._param,
            self._multiplier / other,
        )

    def __neg__(self) -> Parameter:
        return Parameter(
            self._process,
            self._index,
            self._param,
            -self._multiplier,
        )

    def __repr__(self):
        return (
            f"<Ket 'Parameter' param={self._param}, value={self.value},"
            + f" index={self._index}, pid={hex(id(self._process))}>"
        )

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
            available, value = self._process.get_gradient(self._index)
            if available.value:
                self._gradient = value.value
        return self._gradient
