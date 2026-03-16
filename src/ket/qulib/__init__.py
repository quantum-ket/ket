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
import os
import random
import sys
from typing import Callable, Literal
from math import sqrt, exp
from collections.abc import Iterable
from inspect import signature
import multiprocessing as mp

from ..clib.libket import BatchExecution
from ..base import Process, Quant
from ..operations import dump, exp_value
from ..gates import H, CNOT, X
from ..expv import Hamiltonian
from . import gates, prepare, oracle, ham

try:
    import google.colab  # pylint: disable=unused-import

    _IN_NOTEBOOK = True
except ImportError:
    try:
        from IPython import get_ipython

        _IN_NOTEBOOK = get_ipython().__class__.__name__ == "ZMQInteractiveShell"

    except ImportError:
        _IN_NOTEBOOK = False

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
    "oracle",
    "prepare",
    "pauli_string",
    "energy",
    "simulated_annealing",
    "exact_solver",
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
    num_qubits: int | list[int] | None = None,
    args: tuple = (),
    *,
    qubits: int | list[int] | None = None,
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
        num_qubits: Number of qubits.
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

    # only qubits or num_qubits can be specified
    if qubits is not None and num_qubits is not None:
        raise ValueError("Only one of 'qubits' or 'num_qubits' can be specified.")
    if qubits is None and num_qubits is None:
        raise ValueError("One of 'qubits' or 'num_qubits' must be specified.")
    if qubits is not None:
        num_qubits = qubits

    if not QISKIT_AVAILABLE:
        raise RuntimeError(
            "Visualization optional dependence are required. Install with: "
            "pip install ket-lang[plot]"
        )

    if not isinstance(num_qubits, Iterable):
        num_qubits = [num_qubits]
    if not isinstance(args, Iterable):
        args = [args]

    names = list(signature(gate).parameters)[len(args) : len(args) + len(num_qubits)]
    if len(names) != len(num_qubits):
        names = [None for _ in range(len(num_qubits))]

    if qpu_size is not None:
        num_qubits = list(num_qubits)
        total = sum(num_qubits)
        if total > qpu_size:
            raise ValueError(
                f"Total number of qubits {sum(num_qubits)} exceeds the QPU size {qpu_size}"
            )
        num_qubits.append(qpu_size - total)
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
        [qpu_size] if coupling_graph is not None else num_qubits,
        ["Q"] if coupling_graph is not None else names,
        qpu if len(qpu) else None,
        keep_order,
    )

    p = Process(device)
    q = [p.alloc(n) for n in (num_qubits if qpu_size is None else num_qubits[:-1])]
    gate(*args, *q)
    p.execute()

    if "output" not in kwargs and _IN_NOTEBOOK:
        kwargs["output"] = "mpl"

    kwargs["style"] = {**DRAW_STYLE, **kwargs.get("style", {})}

    fig = device.circuit.draw(**kwargs, plot_barriers=False)
    if title is not None:
        fig.suptitle(title)
    return fig


def pauli_string(
    hamiltonian: Callable[[Quant], Hamiltonian],
    num_qubits: int,
    reversed_qubits: bool = False,
) -> list[tuple[str, float]]:
    """
    Extracts the Pauli strings from a Hamiltonian.

    This function evaluates a given Hamiltonian over a dynamically allocated
    quantum process and parses its terms into a list of string representations
    (e.g., ``"IXYI"``) alongside their scalar coefficients. Any qubit not explicitly
    acted upon by a Pauli operator in a term defaults to the Identity operator (``"I"``).

    Args:
        hamiltonian: A callable that takes a collection of qubits
            (``Quant``) and returns a ``Hamiltonian`` object.
        num_qubits: The total number of qubits the Hamiltonian acts on.
        reversed_qubits: If True, the order of the allocated qubits is reversed
            before being passed to the `hamiltonian` function. Defaults to False.

    Returns:
        A list of tuple:
          - String represents a Pauli product (e.g., ``"ZIZX"``)
            across all ``num_qubits``.
          - Float representing the coefficients.
    """
    process = Process(num_qubits=num_qubits)
    qubits = process.alloc(num_qubits)
    if reversed_qubits:
        qubits = reversed(qubits)
    h = hamiltonian(qubits)

    coef = []
    pauli_str = []

    for term in h.pauli_products:
        p = ["I"] * num_qubits
        for qubit, pauli in term.map.items():
            p[qubit] = pauli
        coef.append(term.coef)
        pauli_str.append("".join(p))

    return list(zip(pauli_str, coef))


def energy(  # pylint: disable=too-many-branches
    hamiltonian: Callable[[Quant], Hamiltonian],
    state: int | list[int] | str,
    num_qubits: int | None = None,
) -> float:
    """Calculates the energy of a Hamiltonian.

    If the Hamiltonian is diagonal in the computational basis, the computation
    time is linear with respect to the number of qubits.

    Args:
        hamiltonian: A function that takes qubits and returns the Hamiltonian to be evaluated.
        state: The initial computational basis state. Can be an integer, a list of integers
            (e.g., ``[1, 0, 1]``), or a binary string (e.g., ``101``).
        num_qubits: The total number of qubits in the system.
            If not specified (None), it will be inferred from the length of `state`.

    Returns:
        The expected value (energy) measured for the Hamiltonian in the given state.
    """

    if isinstance(state, int):
        if state < 0:
            raise ValueError("State cannot be a negative integer.")
        if num_qubits is None:
            num_qubits = max(1, state.bit_length())
        elif state >= (1 << num_qubits):
            raise ValueError(
                f"State integer {state} is too large to be represented by {num_qubits} qubits."
            )

    elif isinstance(state, Iterable):
        if num_qubits is None:
            num_qubits = len(state)
        elif num_qubits != len(state):
            raise ValueError(
                f"num_qubits ({num_qubits}) does not match the length of the state ({len(state)})."
            )

        for val in state:
            if str(val) not in ("0", "1"):
                raise ValueError(
                    f"State iterable must only contain 0s and 1s, found: '{val}'."
                )
    else:
        raise TypeError(
            f"Invalid state type: {type(state).__name__}. Expected int, list[int], or str."
        )

    if num_qubits <= 0:
        raise ValueError(f"num_qubits must be a positive integer, got {num_qubits}.")

    if num_qubits is None:
        num_qubits = len(state)
    if not isinstance(state, Iterable):
        state = f"{state:0{num_qubits}b}"

    process = Process(
        execution="batch",
        simulator="sparse",
        num_qubits=num_qubits,
    )
    qubits = process.alloc(num_qubits)

    X(qubits.at(int(i) for i, s in enumerate(state) if str(s) == "1"))

    h = hamiltonian(qubits)

    return exp_value(h).get()


def _simulated_annealing(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    hamiltonian,
    n_bits,
    initial_temp,
    final_temp,
    alpha,
    seed,
):
    random.seed(seed)

    max_val = (1 << n_bits) - 1
    current_state = random.randint(0, max_val)
    current_energy = energy(hamiltonian, current_state, n_bits)

    best_state = current_state
    best_energy = current_energy

    temp = initial_temp

    while temp > final_temp:
        bit_to_flip = random.randint(0, n_bits - 1)
        neighbor = current_state ^ (1 << bit_to_flip)

        neighbor_energy = energy(hamiltonian, neighbor, n_bits)

        delta_e = neighbor_energy - current_energy

        if delta_e < 0 or random.random() < exp(-delta_e / temp):
            current_state = neighbor
            current_energy = neighbor_energy

            if current_energy < best_energy:
                best_state = current_state
                best_energy = current_energy

        temp *= alpha

    return best_state, best_energy


def _mute_worker():
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, sys.stdout.fileno())
    os.dup2(devnull, sys.stderr.fileno())


def simulated_annealing(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    hamiltonian: Callable[[Quant], Hamiltonian],
    num_qubits: int,
    initial_temp: float = 1_000.0,
    final_temp: float = 0.01,
    cooling_rate: float = 0.99,
    num_evaluations: int | None = None,
) -> tuple[int, float]:
    """Find the ground state of a Hamiltonian.

    This function runs multiple independent simulated annealing searches concurrently
    using multiprocessing. It explores the solution space and returns the result
    from the run that achieved the lowest energy.

    Args:
        hamiltonian: A function that takes qubits and returns the Hamiltonian to be minimized.
        num_qubits: The number of qubits in the quantum system.
        initial_temp: The starting temperature for the annealing
            schedule. Higher values allow more exploration. Defaults to 1000.0.
        final_temp: The temperature at which the annealing process
            terminates. Defaults to 0.01.
        cooling_rate: The decay factor applied to the temperature
            at each step (0 < cooling_rate < 1). Defaults to 0.99.
        num_evaluations: The number of independent annealing runs
            to execute. If None, it defaults to the total number of available CPU cores.

    Returns:
        The result of the simulated annealing run that yielded the lowest energy.
        Typically returns a tuple containing the optimal state and its corresponding
        energy.
    """

    if num_qubits <= 0:
        raise ValueError(f"num_qubits must be a positive integer, got {num_qubits}.")

    if initial_temp <= 0.0:
        raise ValueError(f"initial_temp must be strictly positive, got {initial_temp}.")

    if final_temp < 0.0:
        raise ValueError(f"final_temp must be non-negative, got {final_temp}.")

    if initial_temp <= final_temp:
        raise ValueError(
            f"initial_temp ({initial_temp}) must be strictly greater than "
            f"final_temp ({final_temp}) for cooling to occur."
        )

    if not 0.0 < cooling_rate < 1.0:
        raise ValueError(
            f"cooling_rate must be strictly between 0 and 1, got {cooling_rate}."
        )

    if num_evaluations is not None and (num_evaluations <= 0):
        raise ValueError(
            f"num_evaluations must be a positive integer or None, got {num_evaluations}."
        )

    num_cores = mp.cpu_count()

    if num_evaluations is None:
        num_evaluations = num_cores

    args = [
        (
            hamiltonian,
            num_qubits,
            initial_temp,
            final_temp,
            cooling_rate,
            random.random(),
        )
        for _ in range(num_evaluations)
    ]

    if num_evaluations > 1:
        with mp.Pool(processes=num_cores, initializer=_mute_worker) as pool:
            results = pool.starmap(_simulated_annealing, args)

        return min(results, key=lambda se: se[1])

    return _simulated_annealing(*args[0])


def exact_solver(
    hamiltonian: Callable[[Quant], Hamiltonian],
    num_qubits: int,
) -> tuple[int, float]:
    r"""Finds the exact ground state of a Hamiltonian.

    Note:
        This function evaluates :math:`2^\texttt{num_qubits}` states.
        Because the search space grows exponentially, this solver should
        only be used for small quantum systems (typically < 10 qubits).
        For larger systems, use heuristic methods like :func:`~ket.qulib.simulated_annealing`.

    Args:
        hamiltonian: A function that takes qubits and returns the Hamiltonian to be evaluated.
        num_qubits: The total number of qubits in the quantum system.

    Returns:
        A tuple containing the optimal state (as an integer) and its corresponding
        lowest energy.
    """
    if not isinstance(num_qubits, int) or num_qubits <= 0:
        raise ValueError(f"num_qubits must be a positive integer, got {num_qubits}.")

    return min(
        (
            (state, energy(hamiltonian, state, num_qubits))
            for state in range(1 << num_qubits)
        ),
        key=lambda se: se[1],
    )
