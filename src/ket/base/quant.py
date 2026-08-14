"""Quantum register abstraction.

This module defines the :class:`~ket.base.Quant` class, which represents a
list of qubit indices and serves as the fundamental quantum object in Ket.
"""

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial
from typing import Any
import weakref

from ..clib.libket import HasProcess


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

    def __init__(self, *, qubits: list[int], process, undo=None, source=None):
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

        from ..operations import (  # pylint: disable=import-outside-toplevel,cyclic-import
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
        from ..qint import Qint  # pylint: disable=import-outside-toplevel,cyclic-import

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
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ..qint import Qreal

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

    def copy(self, depends_on: list | None = None):
        """Create a copy of this register in a fresh auxiliary register.

        Allocates a new auxiliary qubit register of the same size and uses
        CNOT gates to copy the state qubit-by-qubit. The copy is wrapped
        in :func:`~ket.operations.undo` so that the auxiliary register is
        automatically uncomputed when the returned object goes out of scope.

        Args:
            depends_on: A list of objects (such as other :class:`~ket.Quant` instances)
                that the new auxiliary register depends on. This dependency prevents early
                uncomputation; for example, if ``b`` depends on ``a``, ``a`` cannot be
                uncomputed before ``b``. Defaults to ``None``.

        Returns:
            A new :class:`~ket.Quant` wrapping
            an auxiliary register that holds a copy of this register's state.
        """
        # pylint: disable=protected-access
        if depends_on is not None:
            depends_on = [self, *depends_on]
        else:
            depends_on = [self]

        other = self.ket_process.alloc_aux(len(self), depends_on=depends_on)

        with self.ket_process.block_builder(append=False) as compute:
            for s, o in zip(self.qubits, other.qubits):
                with self.ket_process.block_builder(control=[s]) as block:
                    block.append_gate("PauliX", o)
            compute.lock_control()

        uncompute = compute.inverse()
        self.ket_process.append_block(compute, check_qubits=False)
        self.ket_process._block_qubits(self.qubits)

        def undo():
            if self.ket_process.status().value.decode("utf-8") == "Terminated":
                return

            self.ket_process._unblock_qubits(self.qubits)
            self.ket_process.append_block(uncompute, check_qubits=False)

        return Quant(
            qubits=other.qubits,
            process=self.ket_process,
            undo=undo,
            source=[other],
        )

    def __repr__(self):
        return f"<Ket 'Quant' {self.qubits} pid={hex(id(self.ket_process))}>"
