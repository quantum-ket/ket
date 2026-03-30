"""Measurement classes.

This module provides classes for storing measurement results.
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0


from ctypes import c_size_t
from typing import Callable, Literal, Any

from ket.expv import Hamiltonian

from .clib.libket import HasProcess
from .base import Quant

try:
    import plotly.graph_objs as go
    import plotly.express as px

    VISUALIZE = True
except ImportError:
    VISUALIZE = False

__all__ = [
    "Samples",
    "Measurement",
]


class Measurement(HasProcess):
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

    def __init__(
        self,
        qubits: Quant,
        postprocessing: Callable[[int], Any] | None = None,
    ):
        super().__init__(ket_process=qubits.ket_process)

        if any(self.ket_process._is_aux(q) for q in qubits.qubits):
            raise ValueError("Auxiliary qubits cannot be measured")

        self.qubits = [qubits.qubits[i : i + 64] for i in range(0, len(qubits), 64)]
        self.size = len(qubits)
        self.indexes = [
            self.ket_process.measure((c_size_t * len(qubit))(*qubit), len(qubit)).value
            for qubit in self.qubits
        ]
        self._value = None
        self.postprocessing = postprocessing

    def _check(self):
        if self._value is None:
            available, values = zip(
                *(self.ket_process.get_measurement(index) for index in self.indexes)
            )
            if all(map(lambda a: a.value, available)):
                self._value = 0
                for value, qubit in zip(values, self.qubits):
                    self._value <<= len(qubit)
                    self._value |= value.value

    @property
    def value(self) -> Any | None:
        """Retrieve the measurement value if available."""
        self._check()
        if self.postprocessing is not None:
            return self.postprocessing(self._value)
        return self._value

    @property
    def raw_value(self) -> int | None:
        """Retrieve the measurement value if available (without postprocessing)."""
        self._check()
        return self._value

    @property
    def bitstring(self) -> str | None:
        """Retrieve the measurement bitstring if available."""
        self._check()
        if self._value is not None:
            return f"{self._value:0{self.size}b}"

        return self._value

    def get(self) -> Any:
        """Retrieve the measurement value.

        If the value is not available, the quantum process will execute to get the result.
        """

        self._check()
        if self._value is None:
            self.ket_process.execute()
        return self.value

    def __repr__(self):
        return (
            f"<Ket 'Measurement' indexes={self.indexes}, "
            f"value={self.value}, pid={hex(id(self.ket_process))}>"
        )


def _check_visualize():
    if not VISUALIZE:
        raise RuntimeError(
            "Visualization optional dependence are required. Install with: "
            "pip install ket-lang[plot]"
        )


class Samples(HasProcess):
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

    def __init__(
        self,
        qubits: Quant,
        shots: int = 2048,
        postprocessing: Callable[[int], Any] | None = None,
    ):
        super().__init__(ket_process=qubits.ket_process)

        self.qubits = qubits.qubits
        self.size = len(qubits)

        if any(self.ket_process._is_aux(q) for q in self.qubits):
            raise ValueError("Auxiliary qubits cannot be measured")

        self.index = self.ket_process.sample(
            (c_size_t * len(self.qubits))(*self.qubits),
            len(self.qubits),
            shots,
        ).value
        self._value = None
        self.shots = shots
        self.postprocessing = postprocessing

    def _check(self):
        if self._value is None:
            (
                available,
                states,
                count,
                size,
            ) = self.ket_process.get_sample(self.index)
            if available.value:
                self._value = dict(zip(states[: size.value], count[: size.value]))

    @property
    def value(self) -> dict[Any, int] | None:
        """Retrieve the measurement samples if available."""
        self._check()
        if self.postprocessing is not None:
            return {
                self.postprocessing(state): count
                for state, count in self._value.items()
            }
        return self._value

    @property
    def raw_value(self) -> dict[int, int] | None:
        """Retrieve the measurement samples if available (without postprocessing)."""
        self._check()
        return self._value

    @property
    def bitstring(self) -> dict[str, int] | None:
        """Retrieve the bitstring samples if available."""
        self._check()
        if self._value is not None:
            return {
                f"{state:0{self.size}b}": count for state, count in self._value.items()
            }
        return self._value

    @property
    def probability(self) -> dict[int, float] | None:
        """Retrieve the measurement probabilities if available."""
        self._check()
        if self._value is not None:
            return {state: count / self.shots for state, count in self._value.items()}
        return self._value

    def get(self) -> dict[int, int]:
        """Retrieve the measurement samples.

        If the value is not available, the quantum process will execute to get the result.
        """

        self._check()
        if self._value is None:
            self.ket_process.execute()
        return self.value

    def most_frequent_state(self) -> int:
        """Retrieve the most frequent state.

        If the value is not available, the quantum process will execute to get the result.
        """

        return max(self.get().items(), key=lambda sc: sc[1])[0]

    def histogram(
        self,
        mode: Literal["bin", "dec"] = "dec",
        data: Literal["probability", "count"] = "count",
        hamiltonian: Callable[[Quant], Hamiltonian] | None = None,
        plot_filter: Callable[[int, float | int], bool] | None = None,
        **kwargs,
    ) -> go.Figure:
        """Generate a histogram representing the sample.

        This method creates a histogram visualizing the sample distribution.

        Note:
            This method requires additional dependencies from ``ket-lang[plot]``.

            Install with: ``pip install ket-lang[plot]``.

        Args:
            mode: If ``"bin"``, display the states in binary format. If ``"dec"``,
                display the states in decimal format. Defaults to ``"dec"``.
            data: Specify whether to plot ``"probability"`` or ``"count"``. Defaults to ``"count"``.
            hamiltonian: Optional function mapping a Quant to a Hamiltonian to calculate energy.
            plot_filter: Optional function to filter the plotted data. Takes a state (int) and
                its value (probability or count, depending on the ``data`` parameter) and returns
                True to keep the state or False to exclude it.
            **kwargs: Additional keyword arguments passed to :func:`plotly.express.bar`.

        Returns:
            Histogram of sample measurement.
        """
        _check_visualize()

        raw_data = self.get()
        if data == "probability":
            metric_data = {
                state: count / self.shots for state, count in raw_data.items()
            }
        else:
            metric_data = raw_data

        if plot_filter is not None:
            filtered_data = {
                state: val
                for state, val in metric_data.items()
                if plot_filter(state, val)
            }
        else:
            filtered_data = metric_data

        states = list(filtered_data.keys())
        values = list(filtered_data.values())

        state_text = (
            list(map(lambda s: f"|{s:0{len(self.qubits)}b}⟩", states))
            if mode == "bin"
            else states
        )

        plot_data = {
            "State": states,
            data.title(): values,
        }

        if hamiltonian is not None:
            from .qulib import (  # pylint: disable=import-outside-toplevel,cyclic-import
                energy,
            )

            plot_data["Energy"] = [
                energy(hamiltonian, state, num_qubits=self.size) for state in states
            ]

        fig = px.bar(
            plot_data,
            x="State",
            y=data.title(),
            text="Energy" if hamiltonian is not None else None,
            **kwargs,
        )

        fig.update_layout(
            xaxis={
                "tickmode": "array",
                "tickvals": states,
                "ticktext": state_text,
            },
            bargap=0.75,
        )

        return fig

    def __repr__(self) -> str:
        return f"<Ket 'Samples' index={self.index}, pid={hex(id(self.ket_process))}>"
