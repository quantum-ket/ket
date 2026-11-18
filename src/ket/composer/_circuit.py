"""Circuit model of the composer: turns a composed circuit into a Ket gate.

This module has no widget dependency, so a circuit saved with
:meth:`~ket.composer.Composer.to_dict` can be turned into a gate with
:func:`~ket.composer.circuit_gate` even when ``anywidget`` is not installed.
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

import ast
import re
from dataclasses import dataclass
from functools import reduce
from inspect import Parameter as _Param, signature
import keyword
import math
from typing import Any, Callable

from ..base import Process, Quant
from ..gates import (
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
    CNOT,
    RXX,
    RYY,
    RZZ,
)
from ..operations import control

__all__ = [
    "GATES",
    "registers_of",
    "Circuit",
    "circuit_gate",
    "circuit_code",
    "circuit_from_gate",
    "Expression",
]


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
    GateInfo(
        "CNOT",
        "CNOT",
        2,
        (),
        "Two-qubit",
        "Controlled NOT: flips the second qubit when the first is |1⟩",
        _const(CNOT),
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


# ``lambda`` alone (not inside another name): a Python keyword, so the
# parameter is named ``lambda_``.
_KEYWORD_LAMBDA = re.compile(r"(?<!\w)lambda(?!\w)")


class Expression:  # pylint: disable=too-few-public-methods
    """An angle expression such as ``"pi/2"``, ``"2*theta"`` or ``"0.3"``.

    Only arithmetic, numbers, ``pi``, ``tau``, ``e`` and the functions of
    :mod:`math` listed in ``_MATH_FUNCTIONS`` are accepted. Any other name is
    a free parameter of the circuit and becomes a keyword argument of the gate:
    Greek letter names (``alpha``, ``beta``... ``omega``) or the letters
    themselves (``α``, ``θ``...) included. ``lambda``, a Python keyword, is
    read as the parameter ``lambda_``.
    """

    def __init__(self, source: str | float | int):
        self.source = _KEYWORD_LAMBDA.sub("lambda_", str(source).strip())
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


_POSITIONAL = (_Param.POSITIONAL_ONLY, _Param.POSITIONAL_OR_KEYWORD)


def _gate_arg_names(gate: Callable | None, count: int) -> list[str]:
    """Names of the first ``count`` positional parameters of ``gate``.

    As ``qulib.draw`` does, the qubit registers take the names of the gate
    parameters; ``q`` (or ``q0``, ``q1``...) when they cannot be found.
    """
    default = ["q"] if count == 1 else [f"q{i}" for i in range(count)]
    if gate is None:
        return default
    try:
        params = [
            p.name
            for p in signature(gate).parameters.values()
            if p.kind in _POSITIONAL
        ]
    except (TypeError, ValueError):
        return default
    if len(params) < count:
        return default
    return params[:count]


def registers_of(
    num_qubits: int | list[int] | dict[str, int], gate: Callable | None = None
) -> list[list]:
    """Qubit registers ``[[name, size], ...]`` of a composer circuit.

    Args:
        num_qubits: Number of qubits; a list of sizes, one register each; or a
            dict from register name to size.
        gate: Gate whose parameter names name the registers when
            ``num_qubits`` is not a dict.
    """
    if isinstance(num_qubits, dict):
        registers = [[str(name), size] for name, size in num_qubits.items()]
    else:
        sizes = [num_qubits] if isinstance(num_qubits, int) else list(num_qubits)
        registers = [list(r) for r in zip(_gate_arg_names(gate, len(sizes)), sizes)]
    _check_registers(registers)
    return registers


# Register names that would shadow a name the generated code uses.
_RESERVED = {"ctrl", FUNCTION_NAME} | set(_MATH_CONSTANTS) | set(_MATH_FUNCTIONS)


def _check_registers(registers: list[list]):
    if not registers:
        raise ValueError("the circuit needs at least one qubit register")
    seen = set()
    for register in registers:
        name, size = register
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError(f"register {name!r} needs a positive number of qubits")
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(f"{name!r} is not a valid register name")
        if name in _RESERVED or name in GATES:
            raise ValueError(f"{name!r} is reserved and cannot name a register")
        if name in seen:
            raise ValueError(f"two registers named {name!r}")
        seen.add(name)


@dataclass
class _Op:
    gate: GateInfo
    col: int
    qubits: list[int]
    params: list[Expression]
    group: str | None = None  # Editor grouping only: no effect on the gate


class Circuit:
    """A validated composer circuit.

    Args:
        data: Circuit dictionary, as returned by
            :meth:`~ket.composer.Composer.to_dict`, with the keys
            ``"num_qubits"``, ``"ops"`` and, optionally, ``"registers"``
            (``[[name, size], ...]``, in order; one register ``q`` by
            default) and ``"params"`` (default values of the free
            parameters). Qubit indices in ``"ops"`` count across registers.
            An op may have a ``"group"`` name: the editor moves, selects and
            copies the ops of a group together; it does not change the gate.
    """

    def __init__(self, data: dict):
        self.num_qubits = int(data["num_qubits"])
        if self.num_qubits < 1:
            raise ValueError("the circuit needs at least one qubit")
        self.registers: list[list] = [
            [str(n), int(s)] for n, s in (data.get("registers") or [["q", self.num_qubits]])
        ]
        _check_registers(self.registers)
        if sum(size for _, size in self.registers) != self.num_qubits:
            raise ValueError("the register sizes do not add up to num_qubits")
        self._where = [
            (index, local)
            for index, (_, size) in enumerate(self.registers)
            for local in range(size)
        ]
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
        clash = set(self.param_names) & {name for name, _ in self.registers}
        if clash:
            raise ValueError(
                f"{sorted(clash)[0]!r} names both a register and an angle parameter"
            )

    def locate(self, qubit: int) -> tuple[int, int]:
        """``(register index, index in the register)`` of a circuit qubit."""
        return self._where[qubit]

    def label(self, qubit: int) -> str:
        """Source of a qubit in the generated code, as ``a[0]``."""
        index, local = self._where[qubit]
        return f"{self.registers[index][0]}[{local}]"

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
        group = raw.get("group")
        if group is not None and not isinstance(group, str):
            raise ValueError(f"{name}: the group must be a string, got {group!r}")
        return _Op(gate, int(raw["col"]), qubits, params, group)

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
    def used_qubits(self) -> list[int]:
        """For each register, the qubits up to the last one a gate uses."""
        used = [0] * len(self.registers)
        for ops in self.columns.values():
            for op in ops:
                for qubit in op.qubits:
                    index, local = self._where[qubit]
                    used[index] = max(used[index], local + 1)
        return used

    def cropped(self) -> dict:
        """The circuit dictionary with each register cropped after its last
        used qubit (keeping at least one qubit per register)."""
        sizes = [max(1, used) for used in self.used_qubits]
        starts = [sum(sizes[:i]) for i in range(len(sizes))]
        ops = []
        for col, col_ops in self.columns.items():
            for op in col_ops:
                qubits = []
                for qubit in op.qubits:
                    index, local = self._where[qubit]
                    qubits.append(starts[index] + local)
                entry = {
                    "gate": op.gate.name,
                    "col": col,
                    "qubits": qubits,
                    "params": [e.source for e in op.params],
                }
                if op.group is not None:
                    entry["group"] = op.group
                ops.append(entry)
        return {
            "num_qubits": sum(sizes),
            "registers": [[name, size] for (name, _), size in zip(self.registers, sizes)],
            "ops": ops,
            "params": dict(self.defaults),
        }

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


def _apply_column(ops: list[_Op], values: dict, pick: Callable, controls):
    def apply_all():
        for op in ops:
            gate = op.gate.factory(*(expr(values) for expr in op.params))
            gate(*(pick([q]) for q in op.qubits))

    if not controls:
        apply_all()
        return
    ctrl_qubits = pick([c for c, _ in controls])
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
    registers = [list(r) for r in circuit.registers]
    needed = circuit.used_qubits

    def gate(*qubits, **params):
        if len(qubits) != len(registers):
            raise TypeError(
                f"{FUNCTION_NAME}() takes {len(registers)} qubit register(s) "
                f"({', '.join(name for name, _ in registers)}), got {len(qubits)}"
            )
        qubits = [
            q if isinstance(q, Quant) else reduce(Quant.__add__, q) for q in qubits
        ]
        # In each register, the qubits after the last used one are optional.
        for (name, size), used, arg in zip(registers, needed, qubits):
            if not used <= len(arg) <= size:
                expected = str(size) if used == size else f"{used} to {size}"
                raise ValueError(
                    f"{FUNCTION_NAME}: register {name!r} takes {expected} "
                    f"qubit(s), got {len(arg)}"
                )

        def pick(indices):
            selected = []
            for index in indices:
                reg, local = circuit.locate(index)
                selected.append(qubits[reg][local])
            return reduce(Quant.__add__, selected)

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
            _apply_column(ops, values, pick, controls)
        return qubits[0] if len(qubits) == 1 else tuple(qubits)

    args = ", ".join(
        [name for name, _ in registers]
        + [f"{n}={defaults[n]!r}" if n in defaults else n for n in names]
    )
    gate.__name__ = gate.__qualname__ = FUNCTION_NAME
    gate.__doc__ = (
        f"{FUNCTION_NAME}({args})\n\n"
        f"Gate made with the Ket composer, on {_registers_text(registers)}."
        + (f"\nParameters: {', '.join(names)}." if names else "")
    )
    return gate


def _registers_text(registers) -> str:
    return ", ".join(
        f"{name} ({size} qubit{'s' if size > 1 else ''})" for name, size in registers
    )


def _quant_code(circuit: Circuit, qubits: list[int]) -> str:
    """Source of several qubits: ``a[0]``, ``a.at([0, 2])`` or ``a[0] + b[1]``."""
    where = [circuit.locate(q) for q in qubits]
    if len(qubits) == 1:
        return circuit.label(qubits[0])
    if len({reg for reg, _ in where}) == 1:
        return f"{circuit.registers[where[0][0]][0]}.at({[local for _, local in where]})"
    return " + ".join(circuit.label(q) for q in qubits)


def _column_code(
    circuit: Circuit, controls, ops: list[_Op], used: set[str]
) -> list[str]:
    """Source lines of one column; adds the Ket names it needs to ``used``."""
    ctrl_quant = None
    if controls:
        used.add("ctrl")
        states = [s for _, s in controls]
        ctrl_quant = _quant_code(circuit, [c for c, _ in controls])
        if not all(states):
            # ``q == state`` controls on that state instead of |1...1>.
            if " + " in ctrl_quant:
                ctrl_quant = f"({ctrl_quant})"
            ctrl_quant += f" == 0b{''.join(map(str, states))}"
    lines = []
    for op in ops:
        used.add(op.gate.name)
        gate = op.gate.name
        if op.params:
            gate += "(" + ", ".join(e.source for e in op.params) + ")"
        targets = ", ".join(circuit.label(qubit) for qubit in op.qubits)
        if ctrl_quant is None:
            lines.append(f"    {gate}({targets})")
        else:
            lines.append(f"    ctrl({ctrl_quant}, {gate})({targets})")
    return lines


def _make_code(circuit: Circuit) -> str:
    names = circuit.param_names
    reg_names = [name for name, _ in circuit.registers]
    args = list(reg_names)
    if names:
        # Keyword-only, as in the composer gate: a parameter without a
        # default can then follow one with a default.
        args.append("*")
    for name in names:
        if name in circuit.defaults:
            args.append(f"{name}={circuit.defaults[name]!r}")
        else:
            args.append(name)

    used = set()
    body = []
    for controls, ops in circuit.steps():
        body += _column_code(circuit, controls, ops, used)
    body.append(f"    return {', '.join(reg_names)}")

    lines = []
    if circuit.math_names:
        lines.append(f"from math import {', '.join(sorted(circuit.math_names))}")
    imports = sorted(used)
    lines.append(f"from ket import {', '.join(imports)}" if imports else "")
    lines = [line for line in lines if line]
    lines += ["", "", f"def {FUNCTION_NAME}({', '.join(args)}):"]
    lines.append(
        f'    """Gate made with the Ket composer, on {_registers_text(circuit.registers)}."""'
    )
    lines += body
    return "\n".join(lines) + "\n"


def circuit_gate(data: dict, upto: int | None = None) -> Callable:
    """Build a Ket gate from a composer circuit dictionary.

    Example:

        .. code-block:: python

            from ket import *
            from ket.composer import circuit_gate

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
            :meth:`~ket.composer.Composer.to_dict`.
        upto: Last column to include; ``None`` includes them all.

    Returns:
        The gate ``gate(*registers, **params)``, with one positional argument
        per qubit register. Free names used in the angle expressions are
        keyword arguments, defaulting to ``data["params"]``.
    """
    return Circuit(data).gate(upto)


def circuit_code(data: dict) -> str:
    """Python source of a composer circuit dictionary as a Ket function."""
    return Circuit(data).code()


# Ket instructions -> composer gates.
_NATIVE = {
    "PauliX": "X",
    "PauliY": "Y",
    "PauliZ": "Z",
    "Hadamard": "H",
    "RotationX": "RX",
    "RotationY": "RY",
    "RotationZ": "RZ",
    "Phase": "P",
}


def _pi_fraction(value: float) -> tuple[int, int] | None:
    """``(k, d)`` with ``value == k*pi/d``, for small ``d``, or ``None``."""
    for den in range(1, 17):
        num = value * den / math.pi
        if abs(num - round(num)) < 1e-9 and round(num) != 0:
            return int(round(num)), den
    return None


def _angle_source(value: float) -> str:
    """Angle as an expression: ``pi/2``, ``-3*pi/4``, or the number."""
    frac = _pi_fraction(value)
    if frac is None:
        return f"{value:.12g}"
    num, den = frac
    sign = "-" if num < 0 else ""
    num = abs(num)
    text = "pi" if num == 1 else f"{num}*pi"
    return f"{sign}{text}" if den == 1 else f"{sign}{text}/{den}"


def _instruction_op(instr: dict) -> tuple[str, list[str]]:
    """Composer gate name and angle expressions of a Ket instruction."""
    raw = instr["gate"]
    if isinstance(raw, str):
        name, value = raw, None
    else:
        name, arg = next(iter(raw.items()))
        if "Ref" in arg:  # A process parameter: use its current value.
            value = arg["Ref"]["value"] * arg["Ref"]["multiplier"]
        else:
            value = arg["Value"]
    if name not in _NATIVE:
        raise ValueError(f"the composer does not support the {name} instruction")
    gate = _NATIVE[name]
    return gate, [] if value is None else [_angle_source(value)]


def circuit_from_gate(
    gate: Callable, num_qubits: int | list[int] | dict[str, int]
) -> dict:
    """Record a Ket gate as a composer circuit dictionary.

    The gate is called on ``num_qubits`` qubits of a batch process that never
    runs; the instructions it adds become composer gates, packed into
    columns as early as possible. Phase gates, S and T included, keep their
    angle as a P gate. Gates such as SWAP or QFT appear as the
    elementary gates Ket decomposes them into, and global phases are dropped.

    Example:

        .. code-block:: python

            from ket import *
            from ket.composer import circuit_from_gate

            def bell(q):
                CNOT(H(q[0]), q[1])

            data = circuit_from_gate(bell, 2)  # One register, named q

            def oracle(inputs, target):
                ctrl(inputs, X)(target)

            data = circuit_from_gate(oracle, [3, 1])  # Registers inputs and target

    Args:
        gate: Gate to record, called with one register per positional
            argument. Use a ``lambda`` or :func:`functools.partial` to fix
            other arguments.
        num_qubits: Number of qubits of the gate; a list of sizes, one
            register per gate argument, named after the gate parameters; or a
            dict from register name to size.

    Returns:
        The circuit dictionary, as :meth:`Composer.to_dict` returns.

    Raises:
        ValueError: If the gate uses auxiliary qubits or an instruction the
            composer does not support.
    """
    registers = registers_of(num_qubits, gate)
    num_qubits = sum(size for _, size in registers)
    # Spare qubits let a gate that allocates more fail with a clear message
    # below. The batch process never runs, so its size costs nothing.
    process = Process(execution="batch", num_qubits=num_qubits + 16)
    gate(*(process.alloc(size) for _, size in registers))

    ops: list[dict] = []
    columns: dict[int, bool] = {}  # column -> has controls
    last = [-1] * num_qubits  # last column that touches each qubit
    for instr in process.gates():
        target, controls = instr["target"], sorted(instr.get("control") or [])
        qubits = controls + [target]
        if any(q >= num_qubits for q in qubits):
            raise ValueError(
                "the gate uses auxiliary qubits, which the composer does not support"
            )
        name, params = _instruction_op(instr)

        # Controls act on every gate in their column: a controlled gate
        # takes a column of its own, and other gates skip controlled columns.
        col = max(last[q] for q in qubits) + 1
        while col in columns and (controls or columns[col]):
            col += 1
        columns[col] = bool(controls)
        for q in qubits:
            last[q] = col
        ops += [{"gate": "CTRL", "col": col, "qubits": [c], "params": []} for c in controls]
        ops.append({"gate": name, "col": col, "qubits": [target], "params": params})

    return {"num_qubits": num_qubits, "registers": registers, "ops": ops, "params": {}}
