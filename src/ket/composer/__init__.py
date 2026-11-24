"""Drag-and-drop quantum circuit composer for Jupyter notebooks.

See :class:`Composer`.
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

import copy
from cmath import phase
import json
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ._circuit import (
    GATES,
    Circuit,
    Expression,
    circuit_gate,
    circuit_code,
    circuit_from_gate,
    registers_of,
)

if TYPE_CHECKING:
    from ..expv import Hamiltonian

try:
    import anywidget
    import traitlets

    _WIDGET_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    _WIDGET_AVAILABLE = False

# Only the Composer class is public; the rest are its implementation.
__all__ = ["Composer"]

_STATIC = Path(__file__).parent / "static"

#: Simulation stops above this number of qubits.
MAX_SIMULATED_QUBITS = 24
#: The editor lists at most this number of basis states.
MAX_SHOWN_STATES = 256

_INSTALL_MSG = (
    "The composer needs the optional dependencies of the 'composer' extra: "
    "pip install ket-lang[composer]"
)


def simulate_circuit(
    data: dict, upto: int | None = None, obs: Callable | None = None
) -> dict:
    """Run a composer circuit on KBW in live execution and summarize the state.

    Args:
        data: Circuit dictionary, as returned by :meth:`Composer.to_dict`.
        upto: Last column to include; ``None`` includes them all.
        obs: Observable, see :func:`bind_observable`. With it, each row of
            ``"states"`` gets a sixth item, the energy
            :math:`\\left<b\\right|H\\left|b\\right>` of its basis state, and the
            result gets the ``"energy"`` of the state (the expected value of
            the observable) and the ``"energy_range"`` ``[min, max]`` of the
            basis states of the state.

    Returns:
        A dictionary with the nonzero ``"states"`` (``[basis, real, imag,
        probability, phase]``, most probable first), the probability of
        measuring :math:`\\left|1\\right>` on each qubit (``"marginals"``), the
        Bloch vector of each qubit (``"bloch"``, see :func:`bloch_vectors`), the
        number of nonzero states (``"num_states"``), and whether ``"states"``
        was truncated.
    """
    # Imported here to keep this module light when only the model is used.
    # pylint: disable=import-outside-toplevel
    from ..base import Process
    from ..operations import dump, exp_value

    # pylint: enable=import-outside-toplevel

    circuit = Circuit(data)
    if circuit.num_qubits > MAX_SIMULATED_QUBITS:
        raise ValueError(f"simulation is limited to {MAX_SIMULATED_QUBITS} qubits")
    gate = circuit.gate(upto)
    process = Process(
        execution="live",
        simulator="sparse",
        num_qubits=circuit.num_qubits,
        precision="double",
    )
    registers = [process.alloc(size) for _, size in circuit.registers]
    gate(*registers)

    hamiltonian = None if obs is None else bind_observable(obs, registers)
    result = _summarize(
        dump(*registers).states,
        circuit.num_qubits,
        None if hamiltonian is None else diagonal_energy(hamiltonian),
    )
    if hamiltonian is not None:
        result["energy"] = float(exp_value(hamiltonian).get())
    return result


def bind_observable(obs: Callable, registers: list) -> Hamiltonian:
    """The observable of a composer, as a Hamiltonian on the given registers.

    Args:
        obs: A function that takes one qubit register per positional argument,
            as the composer gate, and returns a
            :class:`~ket.expv.Hamiltonian` or :class:`~ket.expv.Pauli`. It is
            called inside a ``with obs():`` block, so ``X``, ``Y`` and ``Z``
            build Pauli operators. It may also be an unbound Hamiltonian (made
            with ``with obs(n) as q:``), which is bound to all the qubits of
            the circuit, in order.
        registers: Qubits of each register of the circuit.

    Raises:
        TypeError: If ``obs`` does not give a Hamiltonian or Pauli operator.
    """
    # pylint: disable=import-outside-toplevel
    from ..expv import Hamiltonian, Pauli
    from ..gates import obs as obs_context

    # pylint: enable=import-outside-toplevel
    qubits = registers[0]
    for register in registers[1:]:
        qubits = qubits + register
    if isinstance(obs, (Hamiltonian, Pauli)):
        hamiltonian = obs
    else:
        with obs_context():
            hamiltonian = obs(*registers)
    if not isinstance(hamiltonian, (Hamiltonian, Pauli)):
        raise TypeError(
            "the observable must give a Hamiltonian or a Pauli operator, "
            f"got {type(hamiltonian).__name__}"
        )
    if not hamiltonian.is_bound:
        hamiltonian = hamiltonian.bind(qubits)
    if isinstance(hamiltonian, Pauli):
        hamiltonian = Hamiltonian([hamiltonian], hamiltonian.ket_process)
    return hamiltonian


def diagonal_energy(hamiltonian: Hamiltonian) -> list[tuple[float, list[int]]]:
    """Diagonal of a Hamiltonian as ``[(coefficient, z_qubits), ...]``.

    The energy :math:`\\left<b\\right|H\\left|b\\right>` of a basis state ``b``
    only gets contributions from the terms made of ``Z`` and ``I``: each adds
    its coefficient, negated if an odd number of its ``Z`` qubits are
    :math:`\\left|1\\right>`. Terms with ``X`` or ``Y`` have zero diagonal.
    Qubit indices are the ones of the process, which match the circuit qubits.
    """
    diagonal = []
    for term in hamiltonian.terms:
        if any(pauli in ("X", "Y") for pauli in term.map.values()):
            continue
        z_qubits = sorted(q for q, pauli in term.map.items() if pauli == "Z")
        diagonal.append((complex(term.coef).real, z_qubits))
    return diagonal


def _basis_energy(
    basis: int, num_qubits: int, diagonal: list[tuple[float, list[int]]]
) -> float:
    energy = 0.0
    for coef, z_qubits in diagonal:
        parity = sum((basis >> (num_qubits - 1 - q)) & 1 for q in z_qubits) & 1
        energy += -coef if parity else coef
    return energy


def _summarize(
    states: dict[int, complex],
    num_qubits: int,
    diagonal: list[tuple[float, list[int]]] | None = None,
) -> dict:
    """Nonzero states, most probable first, and P(1) of each qubit.

    With ``diagonal`` (see :func:`diagonal_energy`), each row also has the
    energy of its basis state, and the result their ``"energy_range"``.
    """
    marginals = [0.0] * num_qubits
    rows = []
    for basis, amp in states.items():
        prob = abs(amp) ** 2
        if prob < 1e-12:
            continue
        for qubit in range(num_qubits):
            if (basis >> (num_qubits - 1 - qubit)) & 1:
                marginals[qubit] += prob
        row = [basis, amp.real, amp.imag, prob, phase(amp)]
        if diagonal is not None:
            row.append(_basis_energy(basis, num_qubits, diagonal))
        rows.append(row)
    rows.sort(key=lambda row: (-round(row[3], 9), row[0]))
    result = {
        "states": rows[:MAX_SHOWN_STATES],
        "num_states": len(rows),
        "truncated": len(rows) > MAX_SHOWN_STATES,
        "marginals": [min(1.0, m) for m in marginals],
        "bloch": bloch_vectors(states, num_qubits),
    }
    if diagonal is not None and rows:
        energies = [row[5] for row in rows]
        result["energy_range"] = [min(energies), max(energies)]
    return result


def bloch_vectors(states: dict[int, complex], num_qubits: int) -> list[list[float]]:
    """Bloch vector ``[<X>, <Y>, <Z>]`` of each qubit.

    The vector of a qubit is the one of its reduced density matrix. For an
    entangled qubit, it is shorter than 1 (inside the sphere); for a
    maximally entangled one, as in a Bell pair, it is zero.

    Args:
        states: Amplitudes by basis state, first qubit as the most
            significant bit, as :attr:`~ket.quantumstate.QuantumState.states`.
        num_qubits: Number of qubits of the state.
    """
    vectors = []
    for qubit in range(num_qubits):
        bit = 1 << (num_qubits - 1 - qubit)
        coherence = (
            0j  # <0|rho|1> of the qubit: sum of conj(a_b) * a_(b with the bit set)
        )
        z = 0.0
        for basis, amp in states.items():
            prob = abs(amp) ** 2
            if basis & bit:
                z -= prob
            else:
                z += prob
                partner = states.get(basis | bit)
                if partner is not None:
                    coherence += amp.conjugate() * partner
        vectors.append([2 * coherence.real, 2 * coherence.imag, z])
    return vectors


if _WIDGET_AVAILABLE:

    class Composer(anywidget.AnyWidget):  # pylint: disable=too-many-instance-attributes
        """Drag-and-drop quantum circuit editor for Jupyter notebooks.

        The composer opens an editor in the notebook where gates are dragged
        from a palette onto the qubit wires. The composed circuit is an
        ordinary Ket gate: call the composer itself, ``composer(qubits)``, or
        take a snapshot with :attr:`gate`. With ``simulate=True``, the KBW
        simulator runs the circuit in live execution on every edit, and the
        editor shows the quantum state and a Bloch sphere for each qubit.
        With an observable, ``obs``, the editor also shows the energy of the
        state and of each basis state.

        The composer needs the optional dependencies of the ``composer``
        extra:

        .. code-block:: bash

            pip install ket-lang[composer]

        Display the object (leave it as the last expression of a cell) to
        open the editor:

        - Drag gates from the palette onto the wires. Drag a gate off the
          circuit, or select it and press :kbd:`Delete`, to remove it.
        - Click a gate to edit its angles and qubits. Drag on an empty area,
          or Shift+click, to select several gates, and group, copy and paste
          them.
        - Controls (● on :math:`\\left|1\\right>`, ○ on
          :math:`\\left|0\\right>`) control every gate in their column.
        - Click a column number to see the state after that column, and a
          **+** between two column numbers to insert an empty column there.
        - Save and Load write and read the circuit as a JSON file, relative
          to the kernel's working directory.

        Angles are written as expressions, such as ``pi/4`` or ``2*theta``.
        Any name other than ``pi``, ``tau``, ``e`` and the functions of
        :mod:`math` (``sqrt``, ``sin``, ``cos``...) is a free parameter: it
        becomes a keyword argument of the gate, and a slider in the editor
        sets the value used in the simulation and as the argument's default.

        Example:

            .. code-block:: python

                from ket import *
                from ket.composer import Composer

                composer = Composer(num_qubits=3)
                composer  # Shows the editor; compose the circuit there

            .. code-block:: python

                p = Process()
                q = p.alloc(3)
                composer(q)  # theta=... for each free parameter of the circuit
                print(dump(q).show())

            Start from a file or from an existing gate:

            .. code-block:: python

                composer = Composer(file="bell.json")  # Opens bell.json if it exists

                def bell(q):
                    CNOT(H(q[0]), q[1])

                composer = Composer(num_qubits=2, gate=bell)

            Named qubit registers, one positional argument each:

            .. code-block:: python

                composer = Composer(num_qubits={"inputs": 3, "target": 1})
                composer  # Wires inputs₀..inputs₂ and target₀

                composer(p.alloc(3), p.alloc(1))

            Energy of an observable, such as a MaxCut cost Hamiltonian:

            .. code-block:: python

                def cost(q):
                    return -0.5 * sum(1 - Z(q[i]) * Z(q[i + 1]) for i in range(3))

                composer = Composer(num_qubits=4, obs=cost)

        Args:
            num_qubits: Number of qubits of the circuit; a list of sizes, one
                qubit register each, named after the parameters of ``gate``;
                or a dict from register name to size. The gate of the
                composer takes one positional argument per register. Ignored
                when ``file`` exists: the file sets it.
            simulate: Simulate the circuit with KBW, in live execution, on
                every edit and show the quantum state.
            file: JSON file to open the circuit from, if it exists, and the
                default file of :meth:`save` and of the Save button.
            gate: A Ket gate to start from, called with one register per
                positional argument. Its instructions become the gates of the
                composer; gates such as SWAP or QFT appear as the elementary
                gates Ket decomposes them into, and global phases are dropped.
            obs: Observable whose energy the simulation reports: a function
                that takes the qubit registers, as the gate of the composer,
                and returns a :class:`~ket.expv.Hamiltonian` or
                :class:`~ket.expv.Pauli`. It runs inside a ``with obs():``
                block, so ``X``, ``Y`` and ``Z`` build Pauli operators. An
                unbound Hamiltonian, made with ``with obs(n) as q:``, also
                works; it binds to the qubits of the circuit, in order. The
                editor then shows the expected value of the observable for the
                state, and colors each basis state :math:`\\left|b\\right>` by
                its energy :math:`\\left<b\\right|H\\left|b\\right>` instead of by
                its phase.

        Raises:
            ValueError: If ``file`` exists but has no valid circuit, if
                ``gate`` cannot be recorded, or if both are given.
        """

        _esm = _STATIC / "composer.js"
        _css = _STATIC / "composer.css"

        num_qubits = traitlets.Int(3).tag(sync=True)
        registers = traitlets.List([["q", 3]]).tag(sync=True)
        ops = traitlets.List([]).tag(sync=True)
        params = traitlets.Dict({}).tag(sync=True)
        simulate = traitlets.Bool(True).tag(sync=True)
        sim_column = traitlets.Int(-1).tag(sync=True)

        # Python -> front end.
        catalog = traitlets.List([]).tag(sync=True)
        param_names = traitlets.List([]).tag(sync=True)
        code = traitlets.Unicode("").tag(sync=True)
        error = traitlets.Unicode("").tag(sync=True)
        sim_result = traitlets.Dict({}).tag(sync=True)
        has_obs = traitlets.Bool(False).tag(sync=True)
        file = traitlets.Unicode("circuit.json").tag(sync=True)
        status = traitlets.Unicode("").tag(sync=True)

        def __init__(
            self,
            num_qubits: int | list[int] | dict[str, int] = 3,
            simulate: bool = True,
            file: str | PathLike | None = None,
            gate: Callable | None = None,
            obs: Callable | None = None,
            **kwargs,
        ):
            registers = registers_of(num_qubits, gate)
            self._obs = obs
            super().__init__(
                num_qubits=sum(size for _, size in registers),
                registers=registers,
                simulate=simulate,
                has_obs=obs is not None,
                catalog=[g.to_json() for g in GATES.values()],
                **kwargs,
            )
            self._circuit: Circuit | None = None
            if file is not None and gate is not None:
                raise ValueError("give either 'file' or 'gate', not both")
            if file is not None:
                self.file = str(file)
                if Path(file).exists():
                    self.load(file)
            if gate is not None:
                self.ops = circuit_from_gate(gate, dict(registers))["ops"]
            self.observe(self._fit_registers, names=["num_qubits"])
            self.observe(
                self._update,
                names=[
                    "num_qubits",
                    "registers",
                    "ops",
                    "params",
                    "simulate",
                    "sim_column",
                ],
            )
            self.on_msg(self._on_msg)
            self._update()

        @property
        def obs(self) -> Callable | None:
            """Observable of the simulation, as the ``obs`` argument.

            Set it to change the observable (``None`` removes it); the
            editor updates the energies.
            """
            return self._obs

        @obs.setter
        def obs(self, value: Callable | None):
            self._obs = value
            self.has_obs = value is not None
            self._update()

        def _fit_registers(self, _change=None):
            """Grow or shrink the last registers to match ``num_qubits``."""
            registers = [list(r) for r in self.registers]
            extra = self.num_qubits - sum(size for _, size in registers)
            if extra == 0 or self.num_qubits < 1:
                return
            if extra > 0:
                registers[-1][1] += extra
            else:
                while extra < 0:
                    take = min(
                        -extra, registers[-1][1] - (1 if len(registers) == 1 else 0)
                    )
                    registers[-1][1] -= take
                    extra += take
                    if registers[-1][1] == 0:
                        registers.pop()
            self.registers = registers

        def _on_msg(self, _widget, content, _buffers):
            """Save and Load buttons of the editor."""
            action = content.get("action") if isinstance(content, dict) else None
            if action not in ("save", "load"):
                return
            path = str(content.get("path") or "").strip() or self.file
            try:
                if action == "save":
                    self.save(path)
                    self.status = f"Saved {path}"
                else:
                    self.load(path)
                    self.status = f"Loaded {path}"
            except (OSError, ValueError, KeyError, TypeError) as err:
                self.status = f"Could not {action} {path}: {err}"

        def save(self, file: str | PathLike | None = None):
            """Write the circuit to a JSON file.

            Each register keeps qubits up to the last one that a gate or
            control uses: on 4 qubits, a circuit that only uses ``q[0]`` and
            ``q[2]`` saves as 3 qubits (``q[1]`` stays, ``q[3]`` is cropped).

            Args:
                file: File to write. Defaults to the ``file`` of the
                    constructor, or ``circuit.json``.
            """
            path = Path(file if file is not None else self.file)
            data = Circuit(self.to_dict()).cropped()
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            self.file = str(file if file is not None else self.file)

        def load(self, file: str | PathLike | None = None):
            """Replace the circuit with the one of a JSON file.

            Args:
                file: File to read. Defaults to the ``file`` of the
                    constructor, or ``circuit.json``.

            Raises:
                ValueError: If the file has no valid circuit.
            """
            path = Path(file if file is not None else self.file)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as err:
                raise ValueError(f"{path} is not a JSON file: {err}") from err
            if not isinstance(data, dict):
                raise ValueError(f"{path} has no composer circuit")
            Circuit(data)  # Validate before changing anything.
            with self.hold_trait_notifications():
                self.registers = Circuit(data).registers
                self.num_qubits = int(data["num_qubits"])
                self.ops = copy.deepcopy(list(data.get("ops", [])))
                self.params = {**self.params, **dict(data.get("params") or {})}
                self.sim_column = -1
            self.file = str(file if file is not None else self.file)

        def _update(self, _change=None):
            try:
                circuit = Circuit(self.to_dict())
            except (ValueError, KeyError, TypeError) as err:
                self._circuit = None
                self.error = str(err)
                self.sim_result = {}
                return

            self._circuit = circuit
            self.error = ""
            if self.param_names != circuit.param_names:
                self.param_names = circuit.param_names
            missing = {n: 0.0 for n in circuit.param_names if n not in self.params}
            if missing:
                # Triggers _update again with the new defaults.
                self.params = {**self.params, **missing}
                return
            self.code = circuit.code()

            if not self.simulate:
                self.sim_result = {}
                return
            try:
                self.sim_result = simulate_circuit(
                    self.to_dict(),
                    None if self.sim_column < 0 else self.sim_column,
                    self._obs,
                )
            except Exception as err:  # pylint: disable=broad-except
                self.sim_result = {}
                self.error = f"simulation failed: {err}"

        def to_dict(self) -> dict:
            """The circuit as a dictionary, to save it.

            The dictionary is what :meth:`save` writes.
            """
            used = set()
            for op in self.ops:
                for source in op.get("params") or []:
                    try:
                        used.update(Expression(source).free_names)
                    except ValueError:
                        pass
            return {
                "num_qubits": self.num_qubits,
                "registers": [list(r) for r in self.registers],
                "ops": copy.deepcopy(list(self.ops)),
                "params": {k: v for k, v in self.params.items() if k in used},
            }

        @property
        def gate(self) -> Callable:
            """The composed circuit as a Ket gate.

            The gate is a snapshot: later edits in the editor do not change
            it. Call it as ``gate(*registers, **params)``, with one positional
            argument per qubit register, where ``params`` overrides the slider
            values of the free parameters. Each register may have from the
            number of qubits the circuit uses (up to its last used qubit) to
            its size in the editor.

            Raises:
                ValueError: If the circuit is invalid.
            """
            return Circuit(self.to_dict()).gate()

        def __call__(self, *qubits, **params):
            """Apply the composed circuit, as it is now, to the qubit registers.

            Same as ``composer.gate(*qubits, **params)``, but always uses the
            current circuit of the editor.
            """
            return self.gate(*qubits, **params)

        def to_code(self) -> str:
            """Python source of the circuit as a Ket function."""
            return Circuit(self.to_dict()).code()

        def clear(self):
            """Remove every gate."""
            self.ops = []
            self.sim_column = -1

else:  # pragma: no cover - depends on the environment

    class Composer:  # pylint: disable=too-few-public-methods
        """Placeholder that explains how to install the composer."""

        def __init__(self, *_args, **_kwargs):
            raise ImportError(_INSTALL_MSG)
