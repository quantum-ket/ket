"""Drag-and-drop quantum circuit composer for Jupyter notebooks.

The :class:`Composer` opens an editor in the notebook where gates are dragged
from a palette onto the qubit wires. The composed circuit is returned as an
ordinary Ket gate: call the composer itself, ``composer(qubits)``, or take a
snapshot with :attr:`Composer.gate`. With
``simulate=True``, the KBW simulator runs the circuit in live execution on
every edit and the editor shows the resulting quantum state.

The editor needs the optional dependencies of the ``composer`` extra:

.. code-block:: bash

    pip install ket-lang[composer]

Example:

    .. code-block:: python

        from ket import *
        from ket.qulib.composer import Composer

        composer = Composer(num_qubits=3)
        # or: composer = qulib.composer(num_qubits=3)
        composer  # Shows the editor; compose the circuit there

    .. code-block:: python

        p = Process()
        q = p.alloc(3)
        composer(q)  # theta=... for each free parameter of the circuit
        print(dump(q).show())

Angles are written as expressions, such as ``pi/4`` or ``2*theta``. Any name
other than ``pi``, ``tau``, ``e`` and the functions of :mod:`math` (``sqrt``,
``sin``, ``cos``...) is a free parameter: it becomes a keyword argument of the
gate, and a slider in the editor sets the value used in the simulation and as
the argument's default.

Controls (● on :math:`\\left|1\\right>`, ○ on :math:`\\left|0\\right>`)
control every gate in their column.
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
import sys
import types
from typing import Callable

from ._circuit import GATES, Circuit, Expression, circuit_gate, circuit_code

try:
    import anywidget
    import traitlets

    _WIDGET_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    _WIDGET_AVAILABLE = False

__all__ = ["Composer", "circuit_gate", "circuit_code", "Expression", "GATES"]

_STATIC = Path(__file__).parent / "static"

#: Simulation stops above this number of qubits.
MAX_SIMULATED_QUBITS = 24
#: The editor lists at most this number of basis states.
MAX_SHOWN_STATES = 256

_INSTALL_MSG = (
    "The composer needs the optional dependencies of the 'composer' extra: "
    "pip install ket-lang[composer]"
)


def simulate_circuit(data: dict, upto: int | None = None) -> dict:
    """Run a composer circuit on KBW in live execution and summarize the state.

    Args:
        data: Circuit dictionary, as returned by :meth:`Composer.to_dict`.
        upto: Last column to include; ``None`` includes them all.

    Returns:
        A dictionary with the nonzero ``"states"`` (``[basis, real, imag,
        probability, phase]``, most probable first), the probability of
        measuring :math:`\\left|1\\right>` on each qubit (``"marginals"``), the
        number of nonzero states (``"num_states"``), and whether ``"states"``
        was truncated.
    """
    # Imported here to keep this module light when only the model is used.
    from ...base import Process  # pylint: disable=import-outside-toplevel
    from ...operations import dump  # pylint: disable=import-outside-toplevel

    circuit = Circuit(data)
    if circuit.num_qubits > MAX_SIMULATED_QUBITS:
        raise ValueError(
            f"simulation is limited to {MAX_SIMULATED_QUBITS} qubits"
        )
    gate = circuit.gate(upto)
    process = Process(
        execution="live",
        simulator="sparse",
        num_qubits=circuit.num_qubits,
        precision="double",
    )
    qubits = process.alloc(circuit.num_qubits)
    gate(qubits)

    return _summarize(dump(qubits).states, circuit.num_qubits)


def _summarize(states: dict[int, complex], num_qubits: int) -> dict:
    """Nonzero states, most probable first, and P(1) of each qubit."""
    marginals = [0.0] * num_qubits
    rows = []
    for basis, amp in states.items():
        prob = abs(amp) ** 2
        if prob < 1e-12:
            continue
        for qubit in range(num_qubits):
            if (basis >> (num_qubits - 1 - qubit)) & 1:
                marginals[qubit] += prob
        rows.append([basis, amp.real, amp.imag, prob, phase(amp)])
    rows.sort(key=lambda row: (-round(row[3], 9), row[0]))
    return {
        "states": rows[:MAX_SHOWN_STATES],
        "num_states": len(rows),
        "truncated": len(rows) > MAX_SHOWN_STATES,
        "marginals": [min(1.0, m) for m in marginals],
    }


if _WIDGET_AVAILABLE:

    class Composer(anywidget.AnyWidget):  # pylint: disable=too-many-instance-attributes
        """Drag-and-drop quantum circuit editor for Jupyter notebooks.

        Display the object (leave it as the last expression of a cell) to open
        the editor. Drag gates from the palette onto the wires; drag a gate
        off the circuit, or select it and press :kbd:`Delete`, to remove it.
        Click a gate to edit its angles and qubits. Click a column number to
        see the state after that column. The Save and Load buttons write and
        read the circuit as a JSON file, relative to the kernel's working
        directory.

        Example:

            .. code-block:: python

                from ket.qulib.composer import Composer

                composer = Composer(file="bell.json")  # Opens bell.json if it exists
                composer

        Args:
            num_qubits: Number of qubits of the circuit. Ignored when ``file``
                exists: the file sets it.
            simulate: Simulate the circuit with KBW, in live execution, on
                every edit and show the quantum state.
            file: JSON file to open the circuit from, if it exists, and the
                default file of :meth:`save` and of the Save button.

        Raises:
            ValueError: If ``file`` exists but has no valid circuit.
        """

        _esm = _STATIC / "composer.js"
        _css = _STATIC / "composer.css"

        num_qubits = traitlets.Int(3).tag(sync=True)
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
        file = traitlets.Unicode("circuit.json").tag(sync=True)
        status = traitlets.Unicode("").tag(sync=True)

        def __init__(
            self,
            num_qubits: int = 3,
            simulate: bool = True,
            file: str | PathLike | None = None,
            **kwargs,
        ):
            super().__init__(
                num_qubits=num_qubits,
                simulate=simulate,
                catalog=[g.to_json() for g in GATES.values()],
                **kwargs,
            )
            self._circuit: Circuit | None = None
            if file is not None:
                self.file = str(file)
                if Path(file).exists():
                    self.load(file)
            self.observe(
                self._update,
                names=["num_qubits", "ops", "params", "simulate", "sim_column"],
            )
            self.on_msg(self._on_msg)
            self._update()

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

            The file keeps qubits up to the last one that a gate or control
            uses: on 4 qubits, a circuit that only uses ``q0`` and ``q2``
            saves as 3 qubits (``q1`` stays, ``q3`` is cropped).

            Args:
                file: File to write. Defaults to the ``file`` of the
                    constructor, or ``circuit.json``.
            """
            path = Path(file if file is not None else self.file)
            data = self.to_dict()
            data["num_qubits"] = max(1, Circuit(data).used_qubits)
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
                )
            except Exception as err:  # pylint: disable=broad-except
                self.sim_result = {}
                self.error = f"simulation failed: {err}"

        def to_dict(self) -> dict:
            """The circuit as a dictionary, to save it.

            The dictionary is what :meth:`save` writes, and works with
            :func:`circuit_gate` and :func:`circuit_code`.
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
                "ops": copy.deepcopy(list(self.ops)),
                "params": {k: v for k, v in self.params.items() if k in used},
            }

        @property
        def gate(self) -> Callable:
            """The composed circuit as a Ket gate.

            The gate is a snapshot: later edits in the editor do not change
            it. Call it as ``gate(qubits, **params)``, where ``params``
            overrides the slider values of the free parameters. ``qubits`` may
            have from the number of qubits the circuit uses (up to its last
            used qubit) to :attr:`num_qubits` qubits.

            Raises:
                ValueError: If the circuit is invalid.
            """
            return Circuit(self.to_dict()).gate()

        def __call__(self, qubits, **params):
            """Apply the composed circuit, as it is now, to ``qubits``.

            Same as ``composer.gate(qubits, **params)``, but always uses the
            current circuit of the editor.
            """
            return self.gate(qubits, **params)

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


class _CallableModule(types.ModuleType):  # pylint: disable=too-few-public-methods
    """Let ``qulib.composer(...)`` create a :class:`Composer`."""

    def __call__(self, *args, **kwargs) -> Composer:
        return Composer(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
