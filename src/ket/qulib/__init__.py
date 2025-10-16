"""Quantum library.

Utilities for preparing quantum states and building quantum algorithms.
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from functools import reduce
from operator import add
from typing import Callable, Literal
from math import sqrt
from collections.abc import Iterable
from inspect import signature

from ..clib.libket import BatchExecution
from ..base import Process
from ..operations import dump
from ..gates import H, CNOT
from . import gates, prepare, math, oracle, ham

try:
    import google.colab  # pylint: disable=unused-import

    _in_notebook = True
except ImportError:
    try:
        from IPython import get_ipython

        _in_notebook = get_ipython().__class__.__name__ == "ZMQInteractiveShell"

    except ImportError:
        _in_notebook = False

try:
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit import library

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

__all__ = [
    "draw",
    "dump_matrix",
    "gates",
    "ham",
    "math",
    "oracle",
    "prepare",
]


def dump_matrix(
    gate: Callable,
    num_qubits: int | list[int] = 1,
    args=(),
    process: Process | None = None,
) -> list[list[complex]]:
    """Get the matrix representation of a quantum gate.

    This function calculates the matrix representation of a quantum gate.

    Args:
        gate: Quantum gate operation to obtain the matrix for.
        num_qubits: Number of qubits.
        args: Classical arguments to pass to the gate function.
        process: Quantum process used to generate the matrix.

    Returns:
        Matrix representation of the quantum gate.
    """

    if isinstance(num_qubits, Iterable):
        qubit_args = num_qubits
    else:
        qubit_args = [num_qubits]

    num_qubits = sum(num_qubits) if isinstance(num_qubits, Iterable) else num_qubits

    if process is None:
        process = Process(
            num_qubits=2 * num_qubits,
            execution="batch",
            simulator="sparse",
        )

    mat = [[0.0j for _ in range(2**num_qubits)] for _ in range(2**num_qubits)]

    qubit_args = [process.alloc(n) for n in qubit_args]
    row = reduce(add, qubit_args)
    column = process.alloc(num_qubits)

    H(column)
    CNOT(column, row)

    gate(*args, *qubit_args)

    state = dump(column + row)

    for state, amp in state.get().items():
        column = state >> num_qubits
        row = state & ((1 << num_qubits) - 1)
        mat[row][column] = amp * sqrt(2**num_qubits)

    return mat


DRAW_STYLE = {
    "textcolor": "#0f1f2e",
    "gatetextcolor": "#f4f8ff",
    "subtextcolor": "#f4f8ff",
    "linecolor": "#0f1f2e",
    "creglinecolor": "#6b7682",
    "gatefacecolor": "#0f7cfb",
    "barrierfacecolor": "#c6cdd5",
    "backgroundcolor": "#f4f8ff",
    "margin": [1.5, 0, 0, 0],
    "displaytext": {
        "u1": "P",
        "rx": "RX",
        "ry": "RY",
        "rz": "RZ",
    },
    "displaycolor": {
        "u1": ["#008699", "#f4f8ff"],
        "x": ["#0f7cfb", "#f4f8ff"],
        "y": ["#0c63c9", "#f4f8ff"],
        "z": ["#009cb3", "#f4f8ff"],
        "h": ["#3f96fc", "#f4f8ff"],
        "rx": ["#063264", "#f4f8ff"],
        "ry": ["#094a97", "#f4f8ff"],
        "rz": ["#008699", "#f4f8ff"],
        "cx": ["#0f7cfb", "#f4f8ff"],
        "ccx": ["#0f7cfb", "#f4f8ff"],
        "mcx": ["#0f7cfb", "#f4f8ff"],
        "cy": ["#0c63c9", "#f4f8ff"],
        "cz": ["#009cb3", "#f4f8ff"],
    },
}


class _IBMDeviceForDraw(BatchExecution):
    """IBMDevice use for qulib.draw"""

    def __init__(
        self,
        num_qubits: list[int],
        names: list[str],
        qpu: dict | None = None,
        keep_order: bool = True,
    ):
        super().__init__()
        qubits = [QuantumRegister(n, l) for n, l in zip(num_qubits, names)]
        self.circuit = QuantumCircuit(*qubits)
        self.num_qubits = sum(num_qubits)
        self.qpu = qpu
        self.keep_order = keep_order
        self.last_gate = None

    def clear(self):
        pass

    def submit_execution(self, circuit, _p):
        self.process_instructions(circuit)

    def get_result(self):
        return {
            "measurements": [],
            "exp_values": [],
            "samples": [],
            "dumps": [],
            "gradients": None,
        }

    def pauli_x(self, target, control):
        if self.keep_order and self.last_gate != "X":
            self.circuit.barrier()
        gate = library.XGate()
        if len(control) >= 1:
            gate = gate.control(len(control))
        self.circuit.append(gate, control + [target])
        self.last_gate = "X"

    def pauli_y(self, target, control):
        if self.keep_order and self.last_gate != "Y":
            self.circuit.barrier()
        gate = library.YGate()
        if len(control) >= 1:
            gate = gate.control(len(control))
        self.circuit.append(gate, control + [target])
        self.last_gate = "Y"

    def pauli_z(self, target, control):
        if self.keep_order and self.last_gate != "Z":
            self.circuit.barrier()
        gate = library.ZGate()
        if len(control) >= 1:
            gate = gate.control(len(control))
        self.circuit.append(gate, control + [target])
        self.last_gate = "Z"

    def hadamard(self, target, control):
        if self.keep_order and self.last_gate != "H":
            self.circuit.barrier()
        gate = library.HGate()
        if len(control) >= 1:
            gate = gate.control(len(control))
        self.circuit.append(gate, control + [target])
        self.last_gate = "H"

    def rotation_x(self, target, control, **kwargs):
        if self.keep_order and self.last_gate != "RX":
            self.circuit.barrier()
        gate = library.RXGate(kwargs["Value"])
        if len(control) >= 1:
            gate = gate.control(len(control))
        self.circuit.append(gate, control + [target])
        self.last_gate = "RX"

    def rotation_y(self, target, control, **kwargs):
        if self.keep_order and self.last_gate != "RY":
            self.circuit.barrier()
        gate = library.RYGate(kwargs["Value"])
        if len(control) >= 1:
            gate = gate.control(len(control))
        self.circuit.append(gate, control + [target])
        self.last_gate = "RY"

    def rotation_z(self, target, control, **kwargs):
        if self.keep_order and self.last_gate != "RZ":
            self.circuit.barrier()
        gate = library.RZGate(kwargs["Value"])
        if len(control) >= 1:
            gate = gate.control(len(control))
        self.circuit.append(gate, control + [target])
        self.last_gate = "RZ"

    def phase(self, target, control, **kwargs):
        if self.keep_order and self.last_gate != "P":
            self.circuit.barrier()
        gate = library.PhaseGate(kwargs["Value"])
        if len(control) >= 1:
            gate = gate.control(len(control))
        self.circuit.append(gate, control + [target])
        self.last_gate = "P"

    def connect(
        self,
    ):
        """Configure process"""
        return super().configure(
            num_qubits=self.num_qubits,
            execution_managed_by_target={},
            qpu=self.qpu,
        )


def draw(  # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
    gate: Callable,
    qubits: int | list[int],
    args: tuple = (),
    *,
    qpu_size: int | None = None,
    u4_gate: Literal["CX", "CZ"] | None = None,
    u2_gates: Literal["ZYZ", "RzSx"] | None = None,
    coupling_graph: list[tuple[int, int]] | None = None,
    title: str | None = None,
    keep_order: bool = True,
    **kwargs,
):
    """Draw a quantum gate using Qiskit.

    Note:
        This method requires additional dependencies from ``ket-lang[plot]``.

        Install with: ``pip install ket-lang[plot]``.

    Args:
        gate: Quantum gate function.
        qubits: Number of qubits.
        args: Classical arguments to pass to the gate function.
        qpu_size: Size of the quantum processing unit (QPU).
            If specified, the number of qubits will be adjusted to fit the QPU size.
        u4_gate: Type of U4 gate to use, either "CX" or "CZ".
        u2_gates: Type of U2 gates to use, either "ZYZ" or "RzSx".
        coupling_graph: Coupling graph of the QPU,
            specified as a list of tuples representing connected qubits.
        title: Title for the circuit diagram.
        keep_order: Maintain the gate call order.
        **kwargs: Keyword arguments to pass to the Qiskit drawer.

    Returns:
        Qiskit circuit diagram of the quantum gate.
    """

    if not QISKIT_AVAILABLE:
        raise RuntimeError(
            "Visualization optional dependence are required. Install with: "
            "pip install ket-lang[plot]"
        )

    if not isinstance(qubits, Iterable):
        qubits = [qubits]
    if not isinstance(args, Iterable):
        args = [args]

    names = list(signature(gate).parameters)[len(args) : len(args) + len(qubits)]
    if len(names) != len(qubits):
        names = [None for _ in range(len(qubits))]

    if qpu_size is not None:
        qubits = list(qubits)
        total = sum(qubits)
        if total > qpu_size:
            raise ValueError(
                f"Total number of qubits {sum(qubits)} exceeds the QPU size {qpu_size}"
            )
        qubits.append(qpu_size - total)
        names.append("AUX")

    qpu = {}

    if u4_gate is not None:
        if not u4_gate in ["CX", "CZ"]:
            raise ValueError("u4_gate must be 'CX' or 'CZ'")
        qpu["u4_gate"] = u4_gate
    if u2_gates is not None:
        if not u2_gates in ["ZYZ", "RzSx"]:
            raise ValueError("u2_gates must be 'ZYZ' or 'RzSx'")
        qpu["u2_gates"] = u2_gates
    if coupling_graph is not None:
        qpu["coupling_graph"] = coupling_graph

    device = _IBMDeviceForDraw(
        [qpu_size] if coupling_graph is not None else qubits,
        ["Q"] if coupling_graph is not None else names,
        qpu if len(qpu) else None,
        keep_order,
    )

    p = Process(device)
    q = [p.alloc(n) for n in (qubits if qpu_size is None else qubits[:-1])]
    gate(*args, *q)
    p.execute()

    if "output" not in kwargs and _in_notebook:
        kwargs["output"] = "mpl"

    kwargs["style"] = {**DRAW_STYLE, **kwargs.get("style", {})}

    fig = device.circuit.draw(**kwargs, plot_barriers=False)
    if title is not None:
        fig.suptitle(title)
    return fig
