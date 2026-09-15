"""Quantum library.

Utilities for preparing quantum states and building quantum algorithms.
"""

from __future__ import annotations

# pylint: disable=too-many-lines

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from functools import reduce
from operator import add
import os
import random
from typing import Callable
from math import sqrt, exp
from collections.abc import Iterable
from collections import Counter
from inspect import signature
from itertools import starmap

try:
    import multiprocess as mp

    _MULTIPROCESS_AVAILABLE = True
except ImportError:
    _MULTIPROCESS_AVAILABLE = False


from ..clib.libket.execution import LiveExecution, BatchExecution
from ..base import Process, Quant
from ..operations import dump, exp_value, measure
from ..gates import H, CNOT, X, Z, obs
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
    from qiskit.circuit import library, Gate
    from qiskit.circuit.gate import Gate as QiskitGate

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
    "GateCount",
    "gate_count",
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
            execution="live",
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
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        num_qubits: list[int],
        names: list[str],
        decompose: bool = False,
        keep_order: bool = True,
        coupling_graph: list[tuple[int, int]] | None = None,
        native_gate_set=None,
    ):
        super().__init__()
        if coupling_graph is not None:
            qubits = [QuantumRegister(sum(num_qubits), "q")]
        else:
            qubits = [QuantumRegister(n, l) for n, l in zip(num_qubits, names)]
        self.circuit = QuantumCircuit(*qubits)
        self.num_qubits = sum(num_qubits)
        self.decompose = decompose
        self.keep_order = keep_order
        self.coupling_graph = coupling_graph
        self.native_gate_set = native_gate_set
        self.last_gate = None

    def _add_gate(self, gate_name, target, control, value):
        # pylint: disable=too-many-branches, too-many-statements
        if gate_name == "PauliX":
            if self.keep_order and self.last_gate != "X":
                self.circuit.barrier()
            gate = library.XGate()
            if len(control) >= 1:
                gate = gate.control(len(control))
            self.circuit.append(gate, control + [target])
            self.last_gate = "X"
        elif gate_name == "PauliY":
            if self.keep_order and self.last_gate != "Y":
                self.circuit.barrier()
            gate = library.YGate()
            if len(control) >= 1:
                gate = gate.control(len(control))
            self.circuit.append(gate, control + [target])
            self.last_gate = "Y"
        elif gate_name == "PauliZ":
            if self.keep_order and self.last_gate != "Z":
                self.circuit.barrier()
            gate = library.ZGate()
            if len(control) >= 1:
                gate = gate.control(len(control))
            self.circuit.append(gate, control + [target])
            self.last_gate = "Z"
        elif gate_name == "Hadamard":
            if self.keep_order and self.last_gate != "H":
                self.circuit.barrier()
            gate = library.HGate()
            if len(control) >= 1:
                gate = gate.control(len(control))
            self.circuit.append(gate, control + [target])
            self.last_gate = "H"
        elif gate_name == "RotationX":
            if self.keep_order and self.last_gate != "RX":
                self.circuit.barrier()
            gate = library.RXGate(value)
            if len(control) >= 1:
                gate = gate.control(len(control))
            self.circuit.append(gate, control + [target])
            self.last_gate = "RX"
        elif gate_name == "RotationY":
            if self.keep_order and self.last_gate != "RY":
                self.circuit.barrier()
            gate = library.RYGate(value)
            if len(control) >= 1:
                gate = gate.control(len(control))
            self.circuit.append(gate, control + [target])
            self.last_gate = "RY"
        elif gate_name == "RotationZ":
            if self.keep_order and self.last_gate != "RZ":
                self.circuit.barrier()
            gate = library.RZGate(value)
            if len(control) >= 1:
                gate = gate.control(len(control))
            self.circuit.append(gate, control + [target])
            self.last_gate = "RZ"
        elif gate_name == "Phase":
            if self.keep_order and self.last_gate != "P":
                self.circuit.barrier()
            gate = library.PhaseGate(value)
            if len(control) >= 1:
                gate = gate.control(len(control))
            self.circuit.append(gate, control + [target])
            self.last_gate = "P"

    def _build_circuit(self, gate_list):
        # pylint: disable=too-many-branches, too-many-statements, too-many-nested-blocks
        # pylint: disable=too-many-branches, too-many-statements
        for g in gate_list:
            if g.get("decomposed"):
                for dec_g in g["decomposed"]:
                    if "U" in dec_g:
                        u_info = dec_g["U"]
                        gate_info = u_info[0]
                        target = u_info[1]
                        control = []
                        if isinstance(gate_info, str):
                            gate_name = gate_info
                            value = 0.0
                        else:
                            gate_name = list(gate_info.keys())[0]
                            val_info = gate_info[gate_name]
                            if isinstance(val_info, dict) and "Ref" in val_info:
                                value = val_info["Ref"]["value"]
                            else:
                                value = val_info.get("Value", 0.0)
                        self._add_gate(gate_name, target, control, value)
                    elif "CNOT" in dec_g:
                        cnot_info = dec_g["CNOT"]
                        target = cnot_info[1]
                        control = [cnot_info[0]]
                        self._add_gate("PauliX", target, control, 0.0)
            else:
                gate_info = g["gate"]
                target = g.get("target", 0)
                control = sorted(g.get("control", []))

                if isinstance(gate_info, str):
                    gate_name = gate_info
                    value = 0.0
                else:
                    gate_name = list(gate_info.keys())[0]
                    val_info = gate_info[gate_name]
                    if isinstance(val_info, dict) and "Ref" in val_info:
                        value = val_info["Ref"]["value"]
                    else:
                        value = val_info.get("Value", 0.0)

                self._add_gate(gate_name, target, control, value)

        # pylint: disable=too-many-branches, too-many-statements

    def exp_value_native(
        self, gate_list, hamiltonians
    ):  # pylint: disable=arguments-renamed, redefined-outer-name
        for g in gate_list:
            gate_name = g[0]
            params = g[1]
            qubits = g[2]

            if gate_name.lower() in ("x", "paulix"):
                gate = library.XGate()
            elif gate_name.lower() in ("y", "pauliy"):
                gate = library.YGate()
            elif gate_name.lower() in ("z", "pauliz"):
                gate = library.ZGate()
            elif gate_name.lower() in ("h", "hadamard"):
                gate = library.HGate()
            elif gate_name.lower() in ("rx", "rotationx"):
                gate = library.RXGate(params[0])
            elif gate_name.lower() in ("ry", "rotationy"):
                gate = library.RYGate(params[0])
            elif gate_name.lower() in ("rz", "rotationz"):
                gate = library.RZGate(params[0])
            elif gate_name.lower() in ("p", "phase"):
                gate = library.PhaseGate(params[0])
            elif gate_name.lower() == "sx":
                gate = library.SXGate()
            elif gate_name.lower() == "sxdg":
                gate = library.SXdgGate()
            elif gate_name.lower() == "cz":
                gate = library.CZGate()
            elif gate_name.lower() == "rzz":
                gate = library.RZZGate(params[0])
            elif gate_name.lower() == "ecr":
                gate = library.ECRGate()
            elif gate_name.lower() == "iswap":
                gate = library.iSwapGate()
            elif gate_name.lower() == "ms":

                gate = (
                    library.XXMinusYYGate(params[0])
                    if len(params) > 0
                    else QiskitGate("MS", 2, [])
                )
            elif gate_name.lower() == "cnot":
                gate = library.CXGate()
            else:
                gate = QiskitGate(gate_name, len(qubits), params)

            if self.keep_order and self.last_gate != gate_name:
                self.circuit.barrier()
            self.circuit.append(gate, qubits)
            self.last_gate = gate_name

        return [0.0] * len(hamiltonians)

    def exp_value(
        self, gate_list, hamiltonian_list
    ):  # pylint: disable=arguments-renamed, redefined-outer-name
        self._build_circuit(gate_list)
        return [0.0] * len(hamiltonian_list)

    def sample(
        self, gate_list, qubits_to_sample, shots: int
    ):  # pylint: disable=arguments-renamed, redefined-outer-name
        self._build_circuit(gate_list)
        sample = Gate(
            name=f"sample({shots=})", num_qubits=len(qubits_to_sample), params=[]
        )
        if self.keep_order and self.last_gate != "M":
            self.circuit.barrier()
        self.circuit.append(sample, qubits_to_sample)
        self.last_gate = "M"
        return ([[0]], [0])

    def connect(self):
        """Configure process"""
        return super().configure(
            self.num_qubits,
            decompose=self.decompose,
            coupling_graph=self.coupling_graph,
            native_gate_set=self.native_gate_set,
        )


def draw(  # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
    gate: Callable,
    num_qubits: int | list[int] | None = None,
    args: tuple = (),
    *,
    qubits: int | list[int] | None = None,
    qpu_size: int | None = None,
    decompose: bool = False,
    coupling_graph: list[tuple[int, int]] | None = None,
    native_gate_set=None,
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
        qubits: Alternative to ``num_qubits`` to specify the number of qubits.
        qpu_size: Size of the quantum processing unit (QPU). If specified,
            the number of qubits will be adjusted to fit the QPU size.
        decompose: Whether to decompose the gate into elementary gates.
        title: Title for the circuit diagram.
        keep_order: Maintain the gate call order.
        **kwargs: Keyword arguments to pass to the Qiskit drawer.

    Returns:
        Qiskit circuit diagram of the quantum gate.

    Example:

        .. code-block:: python

            from ket.qulib import draw
            from ket import H, CNOT

            def my_gate(q):
                H(q[0])
                CNOT(q[0], q[1])

            fig = draw(my_gate, 2)
            # fig.show()
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

    device = _IBMDeviceForDraw(
        num_qubits, names, decompose, keep_order, coupling_graph, native_gate_set
    )

    p = Process(device)
    q = [p.alloc(n) for n in (num_qubits if qpu_size is None else num_qubits[:-1])]
    gate(*args, *q)

    with obs():
        exp_value(Z(q[0])).get()

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

    for term in h.terms:
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
        execution="live",
        simulator="sparse",
        num_qubits=num_qubits,
    )
    qubits = process.alloc(num_qubits)

    X(qubits.at([int(i) for i, s in enumerate(state) if str(s) == "1"]))

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
    os.environ["KBW_PB"] = "false"
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


def simulated_annealing(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    hamiltonian: Callable[[Quant], Hamiltonian],
    num_qubits: int,
    initial_temp: float = 1_000.0,
    final_temp: float = 0.01,
    cooling_rate: float = 0.99,
    num_evaluations: int = 16,
    multiprocessing: bool = True,
) -> tuple[int, float]:
    """Find the ground state of a Hamiltonian.

    This function runs multiple independent simulated annealing searches concurrently
    using multiprocessing. It explores the solution space and returns the result
    from the run that achieved the lowest energy.

    Note:
        When multiprocessing is enabled (the default), this method requires the
        ``multiprocess`` library.

        Install with: ``pip install multiprocess``.

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
            to execute. Defaults to 16.
        multiprocessing: Whether to run the evaluations concurrently using the
            `multiprocess` library. Defaults to True.

    Raises:
        ValueError: If any of the numerical parameters (num_qubits, initial_temp,
            final_temp, cooling_rate, num_evaluations) are out of their valid ranges.
        RuntimeError: If `multiprocessing=True` but the `multiprocess` library
            is not installed.

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

    if num_evaluations <= 0:
        raise ValueError(
            f"num_evaluations must be a positive integer, got {num_evaluations}."
        )

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
        if multiprocessing:
            if not _MULTIPROCESS_AVAILABLE:
                raise RuntimeError(
                    "The `multiprocess` library is required when `multiprocessing=True` "
                    "(the default behavior). Please install it using `pip install multiprocess`, "
                    "or set `multiprocessing=False` to run sequentially."
                )

            with mp.Pool() as pool:
                results = pool.starmap(_simulated_annealing, args)
        else:
            results = list(starmap(_simulated_annealing, args))

        return min(results, key=lambda se: se[1])

    return _simulated_annealing(*args[0])


def exact_solver(
    hamiltonian: Callable[[Quant], Hamiltonian],
    num_qubits: int,
) -> tuple[int, float]:
    r"""Finds the exact ground state of a Hamiltonian.

    Note:
        This function evaluates :math:`2^\texttt{num\_qubits}` states.
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


class _GateCountBase:
    """Base class for counting quantum gates and tracking CNOT depth."""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.counts = Counter()
        self.depths = [0] * num_qubits
        self.cnot_depth = 0

    def _update_cnot_depth(self, control: list[int], target: int):
        """Update CNOT depth for the given control and target qubits."""
        new_depth = max(self.depths[target], *(self.depths[c] for c in control)) + 1
        for c in control:
            self.depths[c] = new_depth
        self.depths[target] = new_depth
        self.cnot_depth = max(self.cnot_depth, new_depth)

    def pauli_x(self, target: int, control: list[int]):
        """Count Pauli-X or CNOT / multi-controlled X gate."""
        if control:
            if len(control) == 1:
                self.counts["cnot"] += 1
            else:
                self.counts[f"c{len(control)}x"] += 1
            self._update_cnot_depth(control, target)
        else:
            self.counts["x"] += 1

    def pauli_y(self, target: int, control: list[int]):
        """Count Pauli-Y or multi-controlled Y gate."""
        if control:
            self.counts[f"c{len(control)}y"] += 1
            self._update_cnot_depth(control, target)
        else:
            self.counts["y"] += 1

    def pauli_z(self, target: int, control: list[int]):
        """Count Pauli-Z or multi-controlled Z gate."""
        if control:
            if len(control) == 1:
                self.counts["cz"] += 1
            else:
                self.counts[f"c{len(control)}z"] += 1
            self._update_cnot_depth(control, target)
        else:
            self.counts["z"] += 1

    def hadamard(self, target: int, control: list[int]):
        """Count Hadamard or multi-controlled H gate."""
        if control:
            self.counts[f"c{len(control)}h"] += 1
            self._update_cnot_depth(control, target)
        else:
            self.counts["h"] += 1

    def rotation_x(
        self,
        target: int,
        control: list[int],
        angle: float,  # pylint: disable=unused-argument
    ):
        """Count Rotation-X gate."""
        if control:
            self.counts[f"c{len(control)}rx"] += 1
            self._update_cnot_depth(control, target)
        else:
            self.counts["rx"] += 1

    def rotation_y(
        self,
        target: int,
        control: list[int],
        angle: float,  # pylint: disable=unused-argument
    ):
        """Count Rotation-Y gate."""
        if control:
            self.counts[f"c{len(control)}ry"] += 1
            self._update_cnot_depth(control, target)
        else:
            self.counts["ry"] += 1

    def rotation_z(
        self,
        target: int,
        control: list[int],
        angle: float,  # pylint: disable=unused-argument
    ):
        """Count Rotation-Z gate."""
        if control:
            self.counts[f"c{len(control)}rz"] += 1
            self._update_cnot_depth(control, target)
        else:
            self.counts["rz"] += 1

    def phase(
        self,
        target: int,
        control: list[int],
        angle: float,  # pylint: disable=unused-argument
    ):
        """Count Phase gate."""
        if control:
            self.counts[f"c{len(control)}p"] += 1
            self._update_cnot_depth(control, target)
        else:
            self.counts["p"] += 1


class GateCount(_GateCountBase, LiveExecution):
    """Live Execution simulator that counts the quantum gates and CNOT depth.

    Example:
        .. code-block:: python


            from ket import *
            from ket.qulib import GateCount

            gc = GateCount(3)
            p = Process(gc)
            q = p.alloc(3)
            H(q[0])
            CNOT(q[0], q[1])
            X(q[2])
            CNOT(q[1], q[2])
            H(q)

            # Trigger execution
            m = measure(q)

            print("Gate Counts:", gc.counts)
            print("CNOT Depth:", gc.cnot_depth)


    Args:
        num_qubits: number of qubits.
    """

    def __init__(self, num_qubits: int):
        _GateCountBase.__init__(self, num_qubits)
        LiveExecution.__init__(self)

    def connect(self):
        """Connect to the execution target."""
        return self.configure(self.num_qubits, decompose=True)

    def measure(self, qubits: list[int]) -> int:
        """Measure callback for LiveExecution."""
        return 0


class BatchGateCount(_GateCountBase, BatchExecution):
    """Batch Execution simulator that counts the quantum gates and CNOT depth.

    Similar to :class:`GateCount`, but uses batch execution mode.
    Note that measurement results will not be available until the process is evaluated
    (e.g. by requesting an expectation value or sampling).

    Args:
        num_qubits: number of qubits.
    """

    def __init__(self, num_qubits: int):
        _GateCountBase.__init__(self, num_qubits)
        BatchExecution.__init__(self)

    def connect(self):
        """Connect to the execution target."""
        return self.configure(self.num_qubits, decompose=True)

    def _count_gates(self, gate_list):
        for gate_dict in gate_list:
            if gate_dict.get("decomposed") is not None:
                self._count_gates(
                    [
                        (
                            {
                                "gate": "PauliX",
                                "target": dec["CNOT"][1],
                                "control": [dec["CNOT"][0]],
                            }
                            if "CNOT" in dec
                            else {
                                "gate": dec["U"][0],
                                "target": dec["U"][1],
                                "control": [],
                            }
                        )
                        for dec in gate_dict["decomposed"]
                    ]
                )
                continue

            gate_name = gate_dict.get("gate")
            if gate_name is None:
                continue
            target = gate_dict.get("target")
            ctrl = sorted(gate_dict.get("control", []))
            val = None

            if isinstance(gate_name, dict):
                gate_name, val_dict = next(iter(gate_name.items()))
                val = val_dict.get("Value")

            self._dispatch_gate(gate_name, target, ctrl, val)

    def _dispatch_gate(self, gate_name, target, ctrl, val):
        """Dispatch a single gate instruction to its counting method."""
        if gate_name == "Hadamard":
            self.hadamard(target, ctrl)
        elif gate_name == "PauliX":
            self.pauli_x(target, ctrl)
        elif gate_name == "PauliY":
            self.pauli_y(target, ctrl)
        elif gate_name == "PauliZ":
            self.pauli_z(target, ctrl)
        elif gate_name == "RotationX":
            self.rotation_x(target, ctrl, val)
        elif gate_name == "RotationY":
            self.rotation_y(target, ctrl, val)
        elif gate_name == "RotationZ":
            self.rotation_z(target, ctrl, val)
        elif gate_name == "Phase":
            self.phase(target, ctrl, val)

    def sample(
        self, gates, qubits_to_sample, shots  # pylint: disable=redefined-outer-name
    ):
        self._count_gates(gates)
        return [], []

    def exp_value(
        self, gates, hamiltonian_list  # pylint: disable=redefined-outer-name
    ):
        self._count_gates(gates)
        return [0.0] * len(hamiltonian_list)

    def sample_native(
        self, gates, qubits, shots  # pylint: disable=redefined-outer-name
    ):
        raise NotImplementedError("BatchGateCount does not support native gates")

    def exp_value_native(
        self, gates, hamiltonians  # pylint: disable=redefined-outer-name
    ):
        raise NotImplementedError("BatchGateCount does not support native gates")

    def gradient(self, gates, hamiltonian):  # pylint: disable=redefined-outer-name
        raise NotImplementedError(
            "BatchGateCount does not support gradient calculation"
        )


def gate_count(
    gate: Callable,
    num_qubits: int | list[int] = 1,
    args: tuple = (),
    *,
    qpu_size: int | None = None,
    batch: bool = False,
) -> tuple[Counter, int]:
    """Count the quantum gates and CNOT depth of a quantum gate function.

    This function executes the provided quantum gate operation on a live
    or batch execution environment and tracks the number of times each fundamental
    gate is called, as well as the CNOT depth (ignoring single-qubit gates).
    Multi-qubit gates are decomposed before counting.

    Args:
        gate: Quantum gate function to analyze.
        num_qubits: Number of qubits. Can be a single integer for one
            quantum register or a list of integers for multiple registers.
        args: Classical arguments to pass to the gate function.
        qpu_size: Size of the quantum processing unit (QPU). Defaults
            to ``sum(num_qubits)``.
        batch: If ``True``, use batch execution mode instead of live execution.

    Returns:
        A tuple containing:
          - A ``collections.Counter`` with the tally of each gate type.
          - The CNOT depth of the quantum circuit.

    Raises:
        ValueError: If the total number of qubits exceeds the specified ``qpu_size``,
            or if the ``qpu_size`` is not a positive integer.

    Example:
        .. code-block:: python

            from ket import H, CNOT
            from ket.qulib import gate_count

            def my_gate(q):
                H(q[0])
                CNOT(q[0], q[1])
                CNOT(q[1], q[2])

            counts, depth = gate_count(my_gate, 3)
            print(counts)
            print(depth)
    """

    if not isinstance(num_qubits, Iterable):
        num_qubits = [num_qubits]

    total_qubits = sum(num_qubits)

    if qpu_size is not None:
        if total_qubits > qpu_size:
            raise ValueError(
                f"Total number of qubits {total_qubits} exceeds the QPU size {qpu_size}"
            )
    else:
        qpu_size = total_qubits

    if qpu_size <= 0:
        raise ValueError(f"QPU size must be a positive integer, got {qpu_size}.")

    if batch:
        gc = BatchGateCount(qpu_size)
    else:
        gc = GateCount(qpu_size)

    process = Process(gc)

    qubits = [process.alloc(n) for n in num_qubits]

    gate(*args, *qubits)

    if batch:
        with obs():
            exp_value(Z(qubits[0][0])).get()
    else:
        measure(qubits[0][0])

    return gc.counts, gc.cnot_depth
