"""Circuit model of the composer: turns a composed circuit into a Ket gate.

This module has no widget dependency, so a circuit saved with
:meth:`~ket.qulib.composer.Composer.to_dict` can be turned into a gate with
:func:`~ket.qulib.composer.circuit_gate` even when ``anywidget`` is not installed.
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

import ast
from dataclasses import dataclass
from functools import reduce
import keyword
import math
from typing import Any, Callable

from ...base import Quant
from ...gates import (
    H,
    X,
    Y,
    Z,
    S,
    SD,
    T,
    TD,
    SX,
    RX,
    RY,
    RZ,
    P,
    U3,
    SWAP,
    RXX,
    RYY,
    RZZ,
)
from ...operations import control

__all__ = ["GATES", "Circuit", "circuit_gate", "circuit_code", "Expression"]


@dataclass(frozen=True)
class GateInfo:
    """A gate of the composer palette."""

    name: str
    label: str
    num_qubits: int
    params: tuple[str, ...]
    group: str
    description: str
    factory: Callable[..., Callable] | None

    def to_json(self) -> dict:
        """Gate description sent to the widget front end."""
        return {
            "name": self.name,
            "label": self.label,
            "num_qubits": self.num_qubits,
            "params": list(self.params),
            "group": self.group,
            "description": self.description,
        }


def _const(gate):
    return lambda: gate


_GATE_LIST = [
    GateInfo("H", "H", 1, (), "Clifford", "Hadamard", _const(H)),
    GateInfo("X", "X", 1, (), "Clifford", "Pauli X (NOT)", _const(X)),
    GateInfo("Y", "Y", 1, (), "Clifford", "Pauli Y", _const(Y)),
    GateInfo("Z", "Z", 1, (), "Clifford", "Pauli Z", _const(Z)),
    GateInfo("S", "S", 1, (), "Clifford", "Phase π/2", _const(S)),
    GateInfo("SD", "S†", 1, (), "Clifford", "Phase -π/2", _const(SD)),
    GateInfo("SX", "√X", 1, (), "Clifford", "Square root of X", _const(SX)),
    GateInfo("T", "T", 1, (), "Phase", "Phase π/4", _const(T)),
    GateInfo("TD", "T†", 1, (), "Phase", "Phase -π/4", _const(TD)),
    GateInfo("P", "P", 1, ("theta",), "Phase", "Phase shift P(θ)", P),
    GateInfo("RX", "RX", 1, ("theta",), "Rotation", "X rotation RX(θ)", RX),
    GateInfo("RY", "RY", 1, ("theta",), "Rotation", "Y rotation RY(θ)", RY),
    GateInfo("RZ", "RZ", 1, ("theta",), "Rotation", "Z rotation RZ(θ)", RZ),
    GateInfo(
        "U3",
        "U3",
        1,
        ("theta", "phi", "lambda"),
        "Rotation",
        "U3(θ, φ, λ)",
        U3,
    ),
    GateInfo("SWAP", "×", 2, (), "Two-qubit", "Swap two qubits", _const(SWAP)),
    GateInfo("RXX", "RXX", 2, ("theta",), "Two-qubit", "XX rotation RXX(θ)", RXX),
    GateInfo("RYY", "RYY", 2, ("theta",), "Two-qubit", "YY rotation RYY(θ)", RYY),
    GateInfo("RZZ", "RZZ", 2, ("theta",), "Two-qubit", "ZZ rotation RZZ(θ)", RZZ),
    GateInfo(
        "CTRL",
        "●",
        1,
        (),
        "Control",
        "Control on |1⟩: controls every gate in its column",
        None,
    ),
    GateInfo(
        "ACTRL",
        "○",
        1,
        (),
        "Control",
        "Control on |0⟩: controls every gate in its column",
        None,
    ),
]

GATES: dict[str, GateInfo] = {g.name: g for g in _GATE_LIST}
_CONTROLS = {"CTRL": 1, "ACTRL": 0}

# Names that an angle expression may use without becoming a gate parameter.
_MATH_CONSTANTS = {"pi": math.pi, "tau": math.tau, "e": math.e}
_MATH_FUNCTIONS = {
    name: getattr(math, name)
    for name in (
        "sqrt",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "exp",
        "log",
    )
}
#: Name of the function in the generated code and of the gate.
FUNCTION_NAME = "circuit"
_BIN_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
_UNARY_OPS = (ast.UAdd, ast.USub)


class Expression:  # pylint: disable=too-few-public-methods
    """An angle expression such as ``"pi/2"``, ``"2*theta"`` or ``"0.3"``.

    Only arithmetic, numbers, ``pi``, ``tau``, ``e`` and the functions of
    :mod:`math` listed in ``_MATH_FUNCTIONS`` are accepted. Any other name is
    a free parameter of the circuit and becomes a keyword argument of the gate.
    """

    def __init__(self, source: str | float | int):
        self.source = str(source).strip()
        if not self.source:
            raise ValueError("empty angle expression")
        try:
            tree = ast.parse(self.source, mode="eval")
        except SyntaxError as err:
            raise ValueError(f"invalid angle expression {self.source!r}") from err
        self.free_names: list[str] = []
        self.math_names: set[str] = set()
        self._check(tree.body)
        self._code = compile(tree, "<composer>", "eval")

    def _check(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                node.value, (int, float)
            ):
                raise ValueError(f"invalid constant in {self.source!r}")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, _BIN_OPS):
            self._check(node.left)
            self._check(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, _UNARY_OPS):
            self._check(node.operand)
        elif isinstance(node, ast.Name):
            if node.id in _MATH_CONSTANTS:
                self.math_names.add(node.id)
            elif node.id in _MATH_FUNCTIONS:
                raise ValueError(f"{node.id!r} is a function in {self.source!r}")
            elif node.id not in self.free_names:
                self.free_names.append(node.id)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _MATH_FUNCTIONS
            and not node.keywords
        ):
            self.math_names.add(node.func.id)
            for arg in node.args:
                self._check(arg)
        else:
            raise ValueError(f"unsupported syntax in angle expression {self.source!r}")

    def __call__(self, params: dict[str, Any]) -> Any:
        env = {**_MATH_CONSTANTS, **_MATH_FUNCTIONS}
        env.update({name: params[name] for name in self.free_names})
        return eval(self._code, {"__builtins__": {}}, env)  # pylint: disable=eval-used


@dataclass
class _Op:
    gate: GateInfo
    col: int
    qubits: list[int]
    params: list[Expression]


class Circuit:
    """A validated composer circuit.

    Args:
        data: Circuit dictionary, as returned by
            :meth:`~ket.qulib.composer.Composer.to_dict`, with the keys
            ``"num_qubits"``, ``"ops"`` and, optionally, ``"params"`` (default
            values of the free parameters).
    """

    def __init__(self, data: dict):
        self.num_qubits = int(data["num_qubits"])
        if self.num_qubits < 1:
            raise ValueError("the circuit needs at least one qubit")
        self.defaults: dict[str, float] = {
            k: float(v) for k, v in dict(data.get("params") or {}).items()
        }

        self.columns: dict[int, list[_Op]] = {}
        self.param_names: list[str] = []
        self.math_names: set[str] = set()
        for raw in data.get("ops", []):
            op = self._parse_op(raw)
            self.columns.setdefault(op.col, []).append(op)
        self._check_columns()

    def _parse_op(self, raw: dict) -> _Op:
        name = raw["gate"]
        if name not in GATES:
            raise ValueError(f"unknown gate {name!r}")
        gate = GATES[name]
        qubits = [int(q) for q in raw["qubits"]]
        if len(qubits) != gate.num_qubits:
            raise ValueError(f"{name} needs {gate.num_qubits} qubit(s), got {qubits}")
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"{name} uses qubit {qubits[0]} twice")
        for qubit in qubits:
            if not 0 <= qubit < self.num_qubits:
                raise ValueError(f"{name} on qubit {qubit} is out of range")
        sources = list(raw.get("params") or [])
        if len(sources) != len(gate.params):
            raise ValueError(f"{name} needs {len(gate.params)} parameter(s)")
        params = []
        for source in sources:
            try:
                expr = Expression(source)
            except ValueError as err:
                raise ValueError(f"{name} on qubit {qubits[0]}: {err}") from err
            params.append(expr)
            self.math_names |= expr.math_names
            for free in expr.free_names:
                if free not in self.param_names:
                    if not free.isidentifier() or keyword.iskeyword(free):
                        raise ValueError(f"invalid parameter name {free!r}")
                    self.param_names.append(free)
        return _Op(gate, int(raw["col"]), qubits, params)

    def _check_columns(self):
        for col, ops in self.columns.items():
            used = set()
            for op in ops:
                for qubit in op.qubits:
                    if qubit in used:
                        raise ValueError(
                            f"two gates on qubit {qubit} in column {col}"
                        )
                    used.add(qubit)

    @property
    def used_qubits(self) -> int:
        """Number of qubits up to the last one a gate or control uses."""
        return (
            max(
                (q for ops in self.columns.values() for op in ops for q in op.qubits),
                default=-1,
            )
            + 1
        )

    def steps(self, upto: int | None = None):
        """Yield ``(controls, control_state, ops)`` for each column, in order.

        Args:
            upto: Last column to include; ``None`` includes them all.
        """
        for col in sorted(self.columns):
            if upto is not None and col > upto:
                break
            ops = self.columns[col]
            controls = sorted(
                (op.qubits[0], _CONTROLS[op.gate.name])
                for op in ops
                if op.gate.name in _CONTROLS
            )
            gates = [op for op in ops if op.gate.name not in _CONTROLS]
            if gates:
                yield controls, gates

    def gate(self, upto: int | None = None) -> Callable:
        """Build the Ket gate of this circuit.

        Args:
            upto: Last column to include; ``None`` includes them all.
        """
        return _make_gate(self, upto)

    def code(self) -> str:
        """Python source of the circuit as a Ket function."""
        return _make_code(self)


def _apply_column(ops: list[_Op], values: dict, qubits: Quant, controls):
    def apply_all():
        for op in ops:
            gate = op.gate.factory(*(expr(values) for expr in op.params))
            gate(*(qubits[q] for q in op.qubits))

    if not controls:
        apply_all()
        return
    ctrl_qubits = qubits.at([c for c, _ in controls])
    states = [s for _, s in controls]
    if all(states):
        with control(ctrl_qubits):
            apply_all()
    else:
        # ``q == state`` flips the |0> controls and undoes the flip on exit.
        with ctrl_qubits == _bits_to_int(states) as flipped:
            with control(flipped):
                apply_all()


def _bits_to_int(bits: list[int]) -> int:
    return int("".join(map(str, bits)), 2)


def _make_gate(circuit: Circuit, upto: int | None) -> Callable:
    steps = list(circuit.steps(upto))
    names = list(circuit.param_names)
    defaults = dict(circuit.defaults)
    needed = circuit.used_qubits

    def gate(qubits: Quant, **params) -> Quant:
        if not isinstance(qubits, Quant):
            qubits = reduce(Quant.__add__, qubits)
        # Trailing qubits that no gate uses are optional.
        if not needed <= len(qubits) <= circuit.num_qubits:
            expected = (
                str(needed)
                if needed == circuit.num_qubits
                else f"{needed} to {circuit.num_qubits}"
            )
            raise ValueError(
                f"{FUNCTION_NAME} acts on {expected} qubit(s), got {len(qubits)}"
            )
        unknown = set(params) - set(names)
        if unknown:
            raise TypeError(
                f"{FUNCTION_NAME}() got unexpected parameter(s) {sorted(unknown)}"
            )
        values = {}
        for name in names:
            if name in params:
                values[name] = params[name]
            elif name in defaults:
                values[name] = defaults[name]
            else:
                raise TypeError(f"{FUNCTION_NAME}() missing parameter {name!r}")
        for controls, ops in steps:
            _apply_column(ops, values, qubits, controls)
        return qubits

    signature = ", ".join(
        ["qubits"]
        + [f"{n}={defaults[n]!r}" if n in defaults else n for n in names]
    )
    gate.__name__ = gate.__qualname__ = FUNCTION_NAME
    gate.__doc__ = (
        f"{FUNCTION_NAME}({signature})\n\n"
        f"Gate made with the Ket composer, on {circuit.num_qubits} qubit(s)."
        + (f"\nParameters: {', '.join(names)}." if names else "")
    )
    return gate


def _column_code(controls, ops: list[_Op], used: set[str]) -> list[str]:
    """Source lines of one column; adds the Ket names it needs to ``used``."""
    lines = []
    indent = "    "
    if controls:
        used.add("control")
        ctrl_qubits = [c for c, _ in controls]
        states = [s for _, s in controls]
        ctrl_quant = (
            f"q[{ctrl_qubits[0]}]" if len(ctrl_qubits) == 1 else f"q.at({ctrl_qubits})"
        )
        if not all(states):
            ctrl_quant += f" == 0b{''.join(map(str, states))}"
        lines.append(f"    with control({ctrl_quant}):")
        indent = "        "
    for op in ops:
        used.add(op.gate.name)
        gate = op.gate.name
        if op.params:
            gate += "(" + ", ".join(e.source for e in op.params) + ")"
        targets = ", ".join(f"q[{qubit}]" for qubit in op.qubits)
        lines.append(f"{indent}{gate}({targets})")
    return lines


def _make_code(circuit: Circuit) -> str:
    names = circuit.param_names
    args = ["q"]
    for name in names:
        if name in circuit.defaults:
            args.append(f"{name}={circuit.defaults[name]!r}")
        else:
            args.append(name)

    used = set()
    body = []
    for controls, ops in circuit.steps():
        body += _column_code(controls, ops, used)
    body.append("    return q")

    lines = []
    if circuit.math_names:
        lines.append(f"from math import {', '.join(sorted(circuit.math_names))}")
    imports = sorted(used)
    lines.append(f"from ket import {', '.join(imports)}" if imports else "")
    lines = [line for line in lines if line]
    lines += ["", "", f"def {FUNCTION_NAME}({', '.join(args)}):"]
    lines.append(
        f'    """Gate made with the Ket composer, on {circuit.num_qubits} qubit(s)."""'
    )
    lines += body
    return "\n".join(lines) + "\n"


def circuit_gate(data: dict, upto: int | None = None) -> Callable:
    """Build a Ket gate from a composer circuit dictionary.

    Example:

        .. code-block:: python

            from ket import *
            from ket.qulib.composer import circuit_gate

            bell = circuit_gate({
                "num_qubits": 2,
                "ops": [
                    {"gate": "H", "col": 0, "qubits": [0]},
                    {"gate": "CTRL", "col": 1, "qubits": [0]},
                    {"gate": "X", "col": 1, "qubits": [1]},
                ],
            })

            p = Process()
            q = bell(p.alloc(2))
            print(dump(q).show())

    Args:
        data: Circuit dictionary, as returned by
            :meth:`~ket.qulib.composer.Composer.to_dict`.
        upto: Last column to include; ``None`` includes them all.

    Returns:
        The gate ``gate(qubits, **params)``. Free names used in the angle
        expressions are keyword arguments, defaulting to ``data["params"]``.
    """
    return Circuit(data).gate(upto)


def circuit_code(data: dict) -> str:
    """Python source of a composer circuit dictionary as a Ket function."""
    return Circuit(data).code()
