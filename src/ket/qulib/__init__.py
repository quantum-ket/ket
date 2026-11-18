"""Quantum library: ready-made building blocks for quantum algorithms.

- :mod:`~ket.qulib.prepare`: states such as GHZ, W, Dicke, or any amplitudes.
- :mod:`~ket.qulib.oracle`: phase and XOR oracles, as used by Grover.
- :mod:`~ket.qulib.gates`: gates from a unitary or diagonal matrix.
- :mod:`~ket.qulib.ham`: Hamiltonians for Max-Cut, TSP, knapsack and mixers.
- :mod:`~ket.qulib.qubo`: QUBO models, solved with :func:`exact_solver`,
  :func:`simulated_annealing` or :func:`parallel_tempering`.
- :func:`gate_count`, :func:`draw` and :func:`dump_matrix`: inspect a gate.

.. code-block:: python

    from ket import *

    p = Process()
    q = qulib.prepare.ghz(p.alloc(3))
    print(sample(q).get())  # {0: ~1024, 7: ~1024}
"""

from __future__ import annotations

# pylint: disable=too-many-lines

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from functools import reduce
from operator import add
from ctypes import c_uint8
import json
import random
from typing import Callable
from math import ceil, log, sqrt
from collections.abc import Iterable
from collections import Counter
from inspect import signature


from ..clib.libket import API as libket
from ..clib.libket.execution import LiveExecution, BatchExecution
from ..base import Process, Quant
from ..operations import dump, exp_value, measure
from ..gates import H, CNOT, X, Z, obs
from ..expv import Hamiltonian, Pauli
from . import gates, prepare, oracle, ham, qubo

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
    "qubo",
    "pauli_string",
    "energy",
    "simulated_annealing",
    "parallel_tempering",
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
            (``Quant``) and returns a ``Hamiltonian`` or ``Pauli`` object.
        num_qubits: The total number of qubits the Hamiltonian acts on.
        reversed_qubits: If True, the order of the allocated qubits is reversed
            before being passed to the `hamiltonian` function. Defaults to False.

    Returns:
        A list of ``(pauli_string, coefficient)`` tuples, where
        ``pauli_string`` is a Pauli product across all ``num_qubits``
        (e.g., ``"ZIZX"``) and ``coefficient`` is its float coefficient.
    """
    process = Process(num_qubits=num_qubits)
    qubits = process.alloc(num_qubits)
    if reversed_qubits:
        qubits = reversed(qubits)
    h = hamiltonian(qubits)
    if isinstance(h, Pauli):
        h = Hamiltonian([h], h.ket_process)

    coef = []
    pauli_str = []

    for term in h.terms:
        p = ["I"] * num_qubits
        for qubit, pauli in term.map.items():
            p[qubit] = pauli
        coef.append(term.coef)
        pauli_str.append("".join(p))

    return list(zip(pauli_str, coef))


_PAULI_NAME = {"X": "PauliX", "Y": "PauliY", "Z": "PauliZ"}


class _DiagonalHamiltonian:
    """Diagonal part of a Hamiltonian, held by Libket for fast classical evaluation.

    Bit ``i`` of a state is the ``i``-th qubit passed to the Hamiltonian, the
    most significant bit of the integer form.
    """

    def __init__(
        self,
        hamiltonian: Callable[[Quant], Hamiltonian] | Hamiltonian | Pauli,
        num_qubits: int,
    ):
        self._ptr = None
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError(
                f"num_qubits must be a positive integer, got {num_qubits}."
            )

        if isinstance(hamiltonian, (Hamiltonian, Pauli)) and not hamiltonian.is_bound:
            # Built with `with obs(n) as q:`; placeholder i is already bit i.
            h = hamiltonian
            position = {q: q for q in range(num_qubits)}
        else:
            process = Process(execution="batch", num_qubits=num_qubits)
            qubits = process.alloc(num_qubits)
            h = hamiltonian(qubits)
            position = {q: i for i, q in enumerate(qubits.qubits)}

        if isinstance(h, Pauli):
            h = Hamiltonian([h], h.ket_process)

        pauli_strings = []
        coefficients = []
        for term in h.terms:
            coef = complex(term.coef)
            if abs(coef.imag) > 1e-10:
                raise ValueError("Complex coefficients are not supported.")
            try:
                pauli_strings.append(
                    [
                        {"pauli": _PAULI_NAME[p], "qubit": position[q]}
                        for q, p in term.map.items()
                        if p != "I"
                    ]
                )
            except KeyError as error:
                raise ValueError(
                    f"The Hamiltonian uses qubit {error.args[0]}, "
                    f"but num_qubits is {num_qubits}."
                ) from None
            coefficients.append(coef.real)

        data = {"pauli_strings": pauli_strings, "coefficients": coefficients}
        self.num_qubits = num_qubits
        self._ptr = libket["ket_diagonal_hamiltonian_new"](
            json.dumps(data).encode("utf-8"), num_qubits
        )

    def __del__(self):
        if getattr(self, "_ptr", None) is not None:
            libket["ket_diagonal_hamiltonian_delete"](self._ptr)
            self._ptr = None

    def to_bits(self, state: int) -> list[int]:
        """Integer state to bits, most significant bit first."""
        n = self.num_qubits
        return [(state >> (n - 1 - i)) & 1 for i in range(n)]

    @staticmethod
    def to_int(bits) -> int:
        """Bits, most significant bit first, to integer state."""
        state = 0
        for bit in bits:
            state = (state << 1) | int(bit)
        return state

    def energy(self, bits: list[int]) -> float:
        """Energy of a basis state given as bits."""
        n = self.num_qubits
        return libket["ket_diagonal_hamiltonian_energy"](
            self._ptr, (c_uint8 * n)(*bits), n
        ).value

    def parallel_tempering(  # pylint: disable=too-many-arguments
        self,
        *,
        num_replicas,
        num_sweeps,
        min_temp,
        max_temp,
        seed,
        num_threads,
        adaptive_temps=False,
        patience=None,
        num_restarts=1,
    ) -> tuple[int, float]:
        """Run parallel tempering and return ``(state, energy)``."""
        n = self.num_qubits
        state = (c_uint8 * n)()
        result = libket["ket_parallel_tempering"](
            self._ptr,
            num_replicas,
            num_sweeps,
            0.0 if min_temp is None else float(min_temp),
            0.0 if max_temp is None else float(max_temp),
            seed,
            num_threads,
            adaptive_temps,
            0 if patience is None else patience,
            num_restarts,
            state,
            n,
        )
        return self.to_int(state), result.value

    def exact_ground_state(self, num_threads: int) -> tuple[int, float]:
        """Check every basis state and return ``(state, energy)`` of a ground state."""
        n = self.num_qubits
        state = (c_uint8 * n)()
        result = libket["ket_exact_ground_state"](self._ptr, num_threads, state, n)
        return self.to_int(state), result.value


def energy(
    hamiltonian: Callable[[Quant], Hamiltonian],
    state: int | list[int] | str,
    num_qubits: int | None = None,
) -> float:
    """Calculates the energy of a Hamiltonian on a computational basis state.

    Computes :math:`\\langle s|H|s\\rangle`. Only the terms made of :math:`Z`
    (and :math:`I`) operators contribute; terms with :math:`X` or :math:`Y`
    have zero diagonal. The evaluation runs in Libket and takes time linear in
    the size of the Hamiltonian, with no quantum simulation.

    For many evaluations of the same Hamiltonian, prefer
    :func:`exact_solver` or :func:`parallel_tempering`, which build the
    Hamiltonian only once.

    Args:
        hamiltonian: A function that takes qubits and returns the Hamiltonian to be
            evaluated, such as a :class:`~ket.qulib.qubo.QUBO`, or an unbound
            Hamiltonian built with ``with obs(n) as q:``.
        state: The computational basis state. Can be an integer (qubit 0 is the
            most significant bit), a list of integers (e.g., ``[1, 0, 1]``), or a
            binary string (e.g., ``"101"``).
        num_qubits: The total number of qubits in the system.
            If not specified (None), it will be inferred from the length of `state`.

    Returns:
        The energy of the Hamiltonian in the given state.
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
        state = list(state)
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

    h = _DiagonalHamiltonian(hamiltonian, num_qubits)
    bits = h.to_bits(state) if isinstance(state, int) else [int(s) for s in state]
    return h.energy(bits)


def parallel_tempering(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    hamiltonian: Callable[[Quant], Hamiltonian],
    num_qubits: int,
    num_replicas: int = 32,
    num_sweeps: int = 1000,
    min_temp: float | None = None,
    max_temp: float | None = None,
    seed: int | None = None,
    num_threads: int | None = None,
    adaptive_temps: bool = False,
    patience: int | None = None,
    num_restarts: int = 1,
) -> tuple[int, float]:
    r"""Find a low-energy basis state of a Hamiltonian with parallel tempering.

    Runs ``num_replicas`` Metropolis chains at temperatures spaced
    geometrically between ``min_temp`` and ``max_temp``. In each sweep, every
    chain tries to flip each bit once; then neighboring chains swap states with
    probability :math:`\min(1, e^{(\beta_i - \beta_j)(E_i - E_j)})`, so good
    states found at high temperature move down to the cold chains. The search
    runs in Libket (Rust), with chains spread over threads.

    Only the diagonal part of the Hamiltonian matters (see :func:`energy`),
    which makes this a solver for QUBO and Ising problems.

    Example:

        .. code-block:: python

            from ket.qulib import parallel_tempering
            from ket.qulib.qubo import QUBO

            qubo = QUBO()
            x = qubo.binary_vars(20)
            qubo += sum((i % 3 - 1) * xi for i, xi in enumerate(x))
            qubo += sum(x) <= 5

            state, energy = parallel_tempering(qubo, qubo.num_qubits, seed=42)

    Args:
        hamiltonian: A function that takes qubits and returns the Hamiltonian to be
            minimized, such as a :class:`~ket.qulib.qubo.QUBO`, or an unbound
            Hamiltonian built with ``with obs(n) as q:``.
        num_qubits: The number of qubits in the system.
        num_replicas: Number of chains (temperatures). Defaults to 32.
        num_sweeps: Number of sweeps. Defaults to 1000.
        min_temp: Coldest temperature. If ``None``, a barrier of size
            :math:`b` is crossed about 0.1% of the time, where :math:`b` is the
            larger of two estimates of the smallest barrier: twice the smallest
            coefficient, and the smallest flip cost found at local minima by a
            few greedy descents.
        max_temp: Hottest temperature. If ``None``, the costliest possible
            flip is accepted about 37% of the time.
        seed: Random seed. Equal seeds give equal results for any number of
            threads. If ``None``, a random seed is used.
        num_threads: Number of threads. If ``None``, uses all cores.
            Small problems run on one thread regardless.
        adaptive_temps: Tune the temperatures during the first quarter of each
            run so that every pair of neighboring replicas swaps equally often.
            ``min_temp`` and ``max_temp`` stay fixed. Defaults to ``False``.
        patience: Stop a run once the best energy has not improved for this
            many sweeps. If ``None`` (default), every run does ``num_sweeps``
            sweeps.
        num_restarts: Number of independent runs; the best result is returned.
            Defaults to 1.

    Returns:
        The best state found, as an integer (qubit 0 is the most significant
        bit), and its energy.
    """
    if not isinstance(num_replicas, int) or num_replicas <= 0:
        raise ValueError(
            f"num_replicas must be a positive integer, got {num_replicas}."
        )
    if not isinstance(num_sweeps, int) or num_sweeps < 0:
        raise ValueError(
            f"num_sweeps must be a non-negative integer, got {num_sweeps}."
        )
    for name, temp in (("min_temp", min_temp), ("max_temp", max_temp)):
        if temp is not None and not temp > 0.0:
            raise ValueError(f"{name} must be strictly positive, got {temp}.")
    if min_temp is not None and max_temp is not None and min_temp > max_temp:
        raise ValueError(
            f"min_temp ({min_temp}) must not be greater than max_temp ({max_temp})."
        )
    if num_threads is not None and (
        not isinstance(num_threads, int) or num_threads <= 0
    ):
        raise ValueError(f"num_threads must be a positive integer, got {num_threads}.")
    if patience is not None and (not isinstance(patience, int) or patience <= 0):
        raise ValueError(f"patience must be a positive integer, got {patience}.")
    if not isinstance(num_restarts, int) or num_restarts <= 0:
        raise ValueError(
            f"num_restarts must be a positive integer, got {num_restarts}."
        )

    h = _DiagonalHamiltonian(hamiltonian, num_qubits)
    return h.parallel_tempering(
        num_replicas=num_replicas,
        num_sweeps=num_sweeps,
        min_temp=min_temp,
        max_temp=max_temp,
        seed=random.getrandbits(64) if seed is None else seed % (1 << 64),
        num_threads=0 if num_threads is None else num_threads,
        adaptive_temps=bool(adaptive_temps),
        patience=patience,
        num_restarts=num_restarts,
    )


def simulated_annealing(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    hamiltonian: Callable[[Quant], Hamiltonian],
    num_qubits: int,
    initial_temp: float = 1_000.0,
    final_temp: float = 0.01,
    cooling_rate: float = 0.99,
    num_evaluations: int = 16,
    multiprocessing: bool = True,  # pylint: disable=unused-argument
) -> tuple[int, float]:
    """Find the ground state of a Hamiltonian.

    Kept for compatibility: runs :func:`parallel_tempering` with temperatures
    from ``final_temp`` to ``initial_temp`` and ``num_evaluations`` replicas
    (at least 2). The number of sweeps is the number of cooling steps,
    :math:`\\log(T_f/T_i)/\\log(\\alpha)`, so the search does at least as much
    work as the old annealing schedule.

    Args:
        hamiltonian: A function that takes qubits and returns the Hamiltonian to be minimized.
        num_qubits: The number of qubits in the quantum system.
        initial_temp: The highest temperature. Defaults to 1000.0.
        final_temp: The lowest temperature. If ``0``, it is chosen
            automatically. Defaults to 0.01.
        cooling_rate: Sets the number of sweeps (0 < cooling_rate < 1).
            Defaults to 0.99.
        num_evaluations: Number of replicas. Defaults to 16.
        multiprocessing: Ignored. The search always runs on Libket threads.

    Raises:
        ValueError: If any of the numerical parameters (num_qubits, initial_temp,
            final_temp, cooling_rate, num_evaluations) are out of their valid ranges.

    Returns:
        The best state found, as an integer, and its energy.
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

    if final_temp > 0.0:
        num_sweeps = ceil(log(final_temp / initial_temp) / log(cooling_rate))
    else:
        num_sweeps = 1000

    return parallel_tempering(
        hamiltonian,
        num_qubits,
        num_replicas=max(2, num_evaluations),
        num_sweeps=max(1, num_sweeps),
        min_temp=final_temp if final_temp > 0.0 else None,
        max_temp=initial_temp,
    )


_EXACT_MAX_QUBITS = 48


def exact_solver(
    hamiltonian: Callable[[Quant], Hamiltonian],
    num_qubits: int,
    num_threads: int | None = None,
) -> tuple[int, float]:
    r"""Finds the exact ground state of a Hamiltonian.

    Checks all :math:`2^\texttt{num\_qubits}` basis states in Libket (Rust),
    in Gray-code order (one bit flip per step) and across threads. Only the
    diagonal part of the Hamiltonian matters (see :func:`energy`). When several
    states share the lowest energy, one of them is returned.

    Note:
        The work doubles with each qubit: about 30 qubits take seconds, 40
        qubits hours. For larger systems, use :func:`~ket.qulib.parallel_tempering`.

    Args:
        hamiltonian: A function that takes qubits and returns the Hamiltonian to be evaluated.
        num_qubits: The total number of qubits, at most 48.
        num_threads: Number of threads. If ``None``, uses all cores.

    Returns:
        A tuple containing the optimal state (as an integer) and its corresponding
        lowest energy.
    """
    if not isinstance(num_qubits, int) or num_qubits <= 0:
        raise ValueError(f"num_qubits must be a positive integer, got {num_qubits}.")
    if num_qubits > _EXACT_MAX_QUBITS:
        raise ValueError(
            f"exact_solver supports at most {_EXACT_MAX_QUBITS} qubits, got {num_qubits}."
        )
    if num_threads is not None and (
        not isinstance(num_threads, int) or num_threads <= 0
    ):
        raise ValueError(f"num_threads must be a positive integer, got {num_threads}.")

    h = _DiagonalHamiltonian(hamiltonian, num_qubits)
    return h.exact_ground_state(0 if num_threads is None else num_threads)


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
        """Count the gates of a batch execution (decomposed when available)."""
        dispatch = self._dispatch_gate
        pauli_x = self.pauli_x
        for gate_dict in gate_list:
            decomposed = gate_dict.get("decomposed")
            if decomposed is not None:
                for dec in decomposed:
                    cnot = dec.get("CNOT")
                    if cnot is not None:
                        pauli_x(cnot[1], [cnot[0]])
                        continue
                    gate_name, target = dec["U"]
                    val = None
                    if isinstance(gate_name, dict):
                        gate_name, val_dict = next(iter(gate_name.items()))
                        val = val_dict.get("Value")
                    dispatch(gate_name, target, [], val)
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

            dispatch(gate_name, target, ctrl, val)

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
    batch: bool = True,
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
        batch: If ``True`` (default), use batch execution: Libket passes the
            whole decomposed circuit in a single call. If ``False``, use live
            execution, with one callback per decomposed gate (slower for large
            circuits). Both modes give the same counts.

    Returns:
        A ``(counts, depth)`` tuple, where ``counts`` is a
        :class:`collections.Counter` with the tally of each gate type and
        ``depth`` is the CNOT depth of the quantum circuit.

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
    for qubit in qubits:
        # The gate is counted for arbitrary inputs, not only for |0...0>.
        process._set_unknown_state(qubit)  # pylint: disable=protected-access

    gate(*args, *qubits)

    if batch:
        with obs():
            exp_value(Z(qubits[0][0])).get()
    else:
        measure(qubits[0][0])

    return gc.counts, gc.cnot_depth
