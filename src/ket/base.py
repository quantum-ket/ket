"""Base quantum programming classes.

This module provides base classes for handling quantum programming in the Ket library. It includes
for handle and store quantum states, measurements.
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0


from ctypes import c_size_t, c_uint8
from json import loads
from typing import Literal, Optional, Any

from .clib.libket import Process as LibketProcess
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
    """Quantum program process.

    A :class:`~ket.base.Process` in Ket handles quantum circuit preparation and execution, serving
    as a direct interface to the Rust library. Its primary usage in quantum programming is to
    allocate qubits using the :meth:`~ket.base.Process.alloc` method.

    Example:

        .. code-block:: python

            from ket import Process

            p = Process()
            qubits = p.alloc(10) # Allocate 10 qubits

    By default, quantum execution is performed by the KBW simulator using sparse mode with 32
    qubits. The KBW simulator in sparse mode handles qubits in a representation equivalent to a
    sparse matrix. This simulator's execution time is related to the amount of superposition in the
    quantum execution, making it suitable as a default when the number of qubits is unknown or the
    quantum state can be represented by a sparse matrix, such as in a GHZ state. The dense simulator
    mode has exponential time complexity with the number of qubits. While it can better explore CPU
    parallelism, the number of qubits must be carefully set. The default number of qubits for the
    dense simulator is 12. The choice of simulator mode depends on the quantum algorithm, as each
    mode has its pros and cons.

    Another parameter for quantum execution on the KBW simulator is between "live" and "batch"
    execution. This configuration determines when quantum instructions will be executed. If set to
    "live", quantum instructions execute immediately after the call. If set to "batch", quantum
    instructions execute only at the end of the process. The default configuration is "live",
    suitable for quantum simulation where the non-clone theorem of quantum mechanics does not need
    to be respected. Batch execution is closer to what is expected from a quantum computer and is
    recommended when preparing quantum code to execute on a QPU.

    Example:

        Batch execution:

        .. code-block:: python

            from ket import *

            p = Process(execution="batch")
            a, b = p.alloc(2)

            CNOT(H(a), b) # Bell state preparation

            d = dump(a + b)

            p.execute()
            # The value of `d` will only be available after executing the process

            print(d.show())

            CNOT(a, b)  # This instruction will raise an error since the process
                        # has already executed.

        Live execution:

        .. code-block:: python

            from ket import *

            p = Process(execution="batch")
            a, b = p.alloc(2)

            CNOT(H(a), b) # Bell state preparation

            # The value of the dump is available right after
            print(dump(a + b).show())

            CNOT(a, b)  # This instruction can execute normally
            H(a)

            print(dump(a + b).show())


    Args:
        configuration: Configuration definition for third-party quantum execution.
            Defaults to :obj:`None`.
        num_qubits: Number of qubits for the KBW simulator. If None and ``simulator=="sparse"``,
            defaults to 32; otherwise, defaults to 12.
        simulator: Simulation mode for the KBW simulator. If None, defaults to ``"sparse"``.
        execution: Execution mode for the KBW simulator. If None, defaults to ``"live"``.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        configuration=None,
        num_qubits: Optional[int] = None,
        simulator: Optional[Literal["sparse", "dense", "dense v2"]] = None,
        execution: Optional[Literal["live", "batch"]] = None,
        coupling_graph: Optional[list[tuple[int, int]]] = None,
        gradient: bool = False,
    ):

        if configuration is not None and any(
            map(lambda a: a is not None, [num_qubits, simulator, execution])
        ):
            raise ValueError("Cannot specify arguments if configuration is provided")

        if configuration is not None:
            super().__init__(configuration)
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
                    coupling_graph=coupling_graph,
                    gradient=gradient,
                )
            )

        self._metadata_buffer_size = 512
        self._metadata_buffer = (c_uint8 * self._metadata_buffer_size)()
        self._instructions_buffer_size = 2048
        self._instructions_buffer = (c_uint8 * self._instructions_buffer_size)()

    def alloc(self, num_qubits: int = 1) -> Quant:
        """Allocate a specified number of qubits and return a :class:`~ket.base.Quant` object.

        Args:
            num_qubits: The number of qubits to allocate. Defaults to 1.

        Returns:
            A list like object representing the allocated qubits.


        Example:
            >>> from ket import Process
            >>> p = Process()
            >>> qubits = p.alloc(3)
            >>> print(qubits)
            <Ket 'Quant' [0, 1, 2] pid=0x...>

        The :meth:`~ket.base.Process.alloc` method allocates the specified number of qubits and
        returns a :class:`~ket.base.Quant` object. Each qubit is assigned a unique index, and the
        resulting :class:`~ket.base.Quant` object encapsulates the allocated qubits along with a
        reference to the parent :class:`~ket.base.Process` object.
        """

        if num_qubits < 1:
            raise ValueError("Cannot allocate less than 1 qubit")

        qubits_index = [self.allocate_qubit().value for _ in range(num_qubits)]
        return Quant(qubits=qubits_index, process=self)

    def _get_ket_process(self):
        return self

    def get_instructions(self) -> list[dict[str, Any]]:
        """Retrieve quantum instructions from the quantum process.

        Returns:
            A list of dictionaries containing quantum instructions extracted from the process.

        The :meth:`~ket.base.Process.get_instructions` method retrieves the quantum instructions
        from the prepared quantum process. It internally calls the
        :meth:`~ket.base.Process.instructions_json` method to obtain the instructions fom the Rust
        library in JSON format, converts the byte data into a list of dictionaries, and returns the
        result.

        Note:
            Ensure that the quantum process has been appropriately prepared for execution using the
            :meth:`~ket.base.Process.prepare_for_execution` method before calling this method. The
            returned instructions provide insights into the quantum circuit and can be useful for
            debugging or analysis purposes.

        Example:
            >>> from ket import *
            >>> p = Process()
            >>> a, b = p.alloc(2)
            >>> CNOT(H(a), b)
            >>> # Get quantum instructions
            >>> instructions = p.get_instructions()
            >>> pprint(instructions)
            [{'Alloc': {'target': 0}},
             {'Alloc': {'target': 1}},
             {'Gate': {'control': [], 'gate': 'Hadamard', 'target': 0}},
             {'Gate': {'control': [0], 'gate': 'PauliX', 'target': 1}}]
        """
        write_size = self.instructions_json(
            self._instructions_buffer, self._instructions_buffer_size
        )
        if write_size.value > self._instructions_buffer_size:
            self._instructions_buffer_size = write_size.value + 1
            self._instructions_buffer = (c_uint8 * self._instructions_buffer_size)()
            return self.get_instructions()

        return loads(bytearray(self._instructions_buffer[: write_size.value]))

    def get_isa_instructions(self) -> list[dict[str, Any]] | None:
        """Retrieve transpiled quantum instructions from the quantum process.

        Returns:
            A list of dictionaries containing quantum instructions extracted from the process
            if the process has been transpiled, otherwise None.

        """
        write_size = self.isa_instructions_json(
            self._instructions_buffer, self._instructions_buffer_size
        )
        if write_size.value > self._instructions_buffer_size:
            self._instructions_buffer_size = write_size.value + 1
            self._instructions_buffer = (c_uint8 * self._instructions_buffer_size)()
            return self.get_isa_instructions()

        return loads(bytearray(self._instructions_buffer[: write_size.value]))

    def get_metadata(self) -> dict[str, Any]:
        """Retrieve metadata from the quantum process.

        Returns:
            A dictionary containing metadata information extracted from the process.

        The :meth:`~ket.base.Process.get_metadata` method retrieves metadata from the prepared
        quantum process. It internally calls the :meth:`~ket.base.Process.metadata_json` method to
        obtain the metadata in JSON format, converts the byte data into a dictionary, and returns
        the result.

        Note:
            Ensure that the quantum process has been appropriately prepared for execution using the
            :meth:`~ket.base.Process.prepare_for_execution` method before calling this method. The
            returned metadata provides information about the quantum circuit execution, including
            depth, gate count, qubit simultaneous operations, status, execution time, and timeout.

        Example:

            >>> from ket import Process
            >>> p = Process()
            >>> a, b = p.alloc(2)
            >>> CNOT(H(a), b)
            >>> # Get metadata
            >>> metadata = p.get_metadata()
            >>> pprint(metadata)
            {'depth': 2,
             'execution_time': None,
             'gate_count': {'1': 1, '2': 1},
             'qubit_simultaneous': 2,
             'status': 'Live',
             'timeout': None}
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
            Parameter(process=self, index=self.set_parameter(p), value=p) for p in param
        ]
        if len(param) == 1:
            return parameters[0]
        return parameters

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

    def histogram(self, **kwargs) -> go.Figure:
        """Generate a histogram representing the sample.

        This method creates a histogram visualizing the probability distribution
        of the sample.

        Note:
            This method requires additional dependencies from ``ket-lang[visualization]``.

            Install with: ``pip install ket-lang[visualization]``.

        Returns:
            Histogram of sample measurement.
        """
        _check_visualize()

        data = {
            "State": list(self.get().keys()),
            "Count": list(self.get().values()),
        }

        fig = px.bar(
            data,
            x="State",
            y="Count",
            **kwargs,
        )

        return fig

    def __repr__(self) -> str:
        return f"<Ket 'Samples' index={self.index}, pid={hex(id(self.process))}>"


def _check_visualize():
    if not VISUALIZE:
        raise RuntimeError(
            "Visualization optional dependence are required. Install with: "
            "pip install ket-lang[visualization]"
        )


class Parameter:
    """Parameter for gradient calculation.

    This class represents a parameter for gradient calculation in a quantum process. It should not
    be instanced directly, but rather obtained from the :meth:`~ket.base.Process.param`
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
