# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

import cmath
import math

import pytest

from ket import Process, dump, H, RY, RX, RZ, P, S, T, Z, SWAP, X, CNOT, QFT, adj, ctrl, control
from ket.composer import circuit_gate, circuit_code, circuit_from_gate, Circuit
from ket.composer import simulate_circuit

GHZ = {
    "num_qubits": 3,
    "ops": [
        {"gate": "H", "col": 0, "qubits": [0]},
        {"gate": "CTRL", "col": 1, "qubits": [0]},
        {"gate": "X", "col": 1, "qubits": [1]},
        {"gate": "CTRL", "col": 2, "qubits": [1]},
        {"gate": "X", "col": 2, "qubits": [2]},
    ],
}

PARAM = {
    "num_qubits": 3,
    "params": {"theta": 0.4},
    "ops": [
        {"gate": "H", "col": 0, "qubits": [0]},
        {"gate": "CTRL", "col": 1, "qubits": [0]},
        {"gate": "X", "col": 1, "qubits": [1]},
        {"gate": "ACTRL", "col": 2, "qubits": [1]},
        {"gate": "RY", "col": 2, "qubits": [2], "params": ["2*theta + pi/8"]},
        {"gate": "SWAP", "col": 3, "qubits": [0, 2], "params": []},
    ],
}


def _reference(q, theta):
    H(q[0])
    ctrl(q[0], X)(q[1])
    with q[1] == 0 as flipped, control(flipped):
        RY(2 * theta + math.pi / 8, q[2])
    SWAP(q[0], q[2])


def _states(gate, *args, **kwargs):
    p = Process()
    q = p.alloc(3)
    gate(q, *args, **kwargs)
    return dump(q).states


def _close(a, b):
    keys = {k for k in a | b if abs(a.get(k, 0)) > 1e-6 or abs(b.get(k, 0)) > 1e-6}
    return all(cmath.isclose(a.get(k, 0), b.get(k, 0), abs_tol=1e-5) for k in keys)


def test_ghz_gate():
    gate = circuit_gate(GHZ)
    assert gate.__name__ == "circuit"
    probs = {k: abs(v) ** 2 for k, v in _states(gate).items()}
    assert set(probs) == {0, 7}
    assert math.isclose(probs[0], 0.5, abs_tol=1e-5)


def test_gate_matches_reference_and_params():
    gate = circuit_gate(PARAM)
    assert _close(_states(gate), _states(_reference, 0.4))
    assert _close(_states(gate, theta=1.1), _states(_reference, 1.1))


def test_generated_code_matches_gate():
    code = circuit_code(PARAM)
    assert "def circuit(q, *, theta=0.4):" in code
    namespace = {}
    exec(code, namespace)  # pylint: disable=exec-used
    assert _close(_states(namespace["circuit"]), _states(circuit_gate(PARAM)))


def test_upto_column():
    gate = circuit_gate(GHZ, upto=1)
    assert set(_states(gate)) == {0, 6}


def test_simulate_circuit():
    result = simulate_circuit(GHZ)
    assert result["num_states"] == 2
    assert [row[0] for row in result["states"]] == [0, 7]
    assert all(math.isclose(m, 0.5, abs_tol=1e-6) for m in result["marginals"])


@pytest.mark.parametrize(
    "ops, message",
    [
        ([{"gate": "FOO", "col": 0, "qubits": [0]}], "unknown gate"),
        ([{"gate": "X", "col": 0, "qubits": [5]}], "out of range"),
        (
            [
                {"gate": "X", "col": 0, "qubits": [0]},
                {"gate": "CTRL", "col": 0, "qubits": [0]},
            ],
            "two gates",
        ),
        ([{"gate": "RX", "col": 0, "qubits": [0], "params": ["__import__('os')"]}], "unsupported"),
        ([{"gate": "RX", "col": 0, "qubits": [0], "params": ["x.real"]}], "unsupported"),
        ([{"gate": "SWAP", "col": 0, "qubits": [1, 1]}], "twice"),
    ],
)
def test_invalid_circuits(ops, message):
    with pytest.raises(ValueError, match=message):
        circuit_gate({"num_qubits": 2, "ops": ops})


def test_gate_argument_errors():
    gate = circuit_gate({**PARAM, "params": {}})
    p = Process()
    with pytest.raises(TypeError, match="missing parameter"):
        gate(p.alloc(3))
    with pytest.raises(TypeError, match="unexpected"):
        gate(p.alloc(3), theta=0.1, phi=0.2)
    with pytest.raises(ValueError, match="register 'q' takes 3 qubit"):
        gate(p.alloc(2), theta=0.1)
    with pytest.raises(ValueError, match="register 'q' takes 3 qubit"):
        gate(p.alloc(4), theta=0.1)


def test_gate_accepts_fewer_qubits_than_the_editor():
    # On 5 qubits, but only q0 and q2 are used: 3 to 5 qubits are accepted.
    gate = circuit_gate(
        {
            "num_qubits": 5,
            "ops": [
                {"gate": "H", "col": 0, "qubits": [0]},
                {"gate": "CTRL", "col": 1, "qubits": [0]},
                {"gate": "X", "col": 1, "qubits": [2]},
            ],
        }
    )
    for size in (3, 4, 5):
        p = Process()
        q = gate(p.alloc(size))
        expected = {0, (0b101 << (size - 3))}
        assert set(dump(q).states) == expected
    p = Process()
    with pytest.raises(ValueError, match="register 'q' takes 3 to 5 qubit"):
        gate(p.alloc(2))
    with pytest.raises(ValueError, match="register 'q' takes 3 to 5 qubit"):
        gate(p.alloc(6))


def test_widget():
    pytest.importorskip("anywidget")
    from ket.composer import Composer  # pylint: disable=import-outside-toplevel

    comp = Composer(num_qubits=3)
    comp.params = dict(PARAM["params"])
    comp.ops = list(PARAM["ops"])
    assert comp.error == ""
    assert comp.param_names == ["theta"]
    assert comp.sim_result["num_states"] > 0
    assert _close(_states(comp.gate), _states(_reference, 0.4))

    assert _close(_states(comp), _states(_reference, 0.4))
    assert _close(_states(comp, theta=0.7), _states(_reference, 0.7))

    snapshot = comp.gate
    comp.params = {"theta": 1.1}
    assert _close(_states(comp.gate), _states(_reference, 1.1))
    assert _close(_states(comp), _states(_reference, 1.1))
    assert _close(_states(snapshot), _states(_reference, 0.4))
    assert "theta=1.1" in comp.to_code()

    comp.ops = comp.ops + [{"gate": "X", "col": 0, "qubits": [0], "params": []}]
    assert "two gates" in comp.error
    with pytest.raises(ValueError):
        comp.gate  # pylint: disable=pointless-statement

    comp.clear()
    assert comp.error == "" and comp.ops == []
    assert circuit_gate(comp.to_dict()).__name__ == "circuit"


def test_composer_exports():
    pytest.importorskip("anywidget")
    import ket  # pylint: disable=import-outside-toplevel
    from ket.composer import Composer  # pylint: disable=import-outside-toplevel

    namespace = {}
    exec("from ket import *", namespace)  # pylint: disable=exec-used
    assert "Composer" not in namespace  # Imported explicitly from ket.composer
    assert not hasattr(ket.qulib, "composer")
    assert ket.composer.__all__ == ["Composer"]
    assert Composer(num_qubits=4).num_qubits == 4


def test_import_ket_does_not_load_the_widget():
    import subprocess  # pylint: disable=import-outside-toplevel
    import sys  # pylint: disable=import-outside-toplevel

    code = "import sys, ket; print('anywidget' in sys.modules, 'ket.composer' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]


def _norm(comp):
    """A composer circuit in a canonical form, to compare two of them."""
    data = Circuit(comp.to_dict()).cropped()
    data["ops"].sort(key=lambda op: (op["col"], op["qubits"]))
    return data


def test_save_and_load(tmp_path):
    pytest.importorskip("anywidget")
    from ket.composer import Composer  # pylint: disable=import-outside-toplevel

    path = tmp_path / "param.json"
    comp = Composer(num_qubits=3, file=path)
    assert comp.ops == [] and comp.file == str(path)  # No file yet: empty
    comp.params = dict(PARAM["params"])
    comp.ops = list(PARAM["ops"])
    comp.save()
    assert path.exists()

    opened = Composer(file=path)  # num_qubits comes from the file
    assert opened.num_qubits == 3
    assert _norm(opened) == _norm(comp)
    assert _close(_states(opened), _states(_reference, 0.4))

    other = Composer(num_qubits=1)
    other.load(path)
    assert _norm(other) == _norm(comp)


def test_save_and_load_buttons(tmp_path):
    pytest.importorskip("anywidget")
    from ket.composer import Composer  # pylint: disable=import-outside-toplevel

    path = str(tmp_path / "ghz.json")
    comp = Composer(num_qubits=3)
    comp.ops = list(GHZ["ops"])
    comp._on_msg(comp, {"action": "save", "path": path}, [])  # pylint: disable=protected-access
    assert comp.status == f"Saved {path}"

    comp.clear()
    comp._on_msg(comp, {"action": "load", "path": path}, [])  # pylint: disable=protected-access
    assert comp.status == f"Loaded {path}"
    assert set(_states(comp)) == {0, 7}

    missing = str(tmp_path / "missing.json")
    comp._on_msg(comp, {"action": "load", "path": missing}, [])  # pylint: disable=protected-access
    assert comp.status.startswith("Could not load")
    assert set(_states(comp)) == {0, 7}  # Unchanged


def test_load_invalid_file(tmp_path):
    pytest.importorskip("anywidget")
    from ket.composer import Composer  # pylint: disable=import-outside-toplevel

    path = tmp_path / "bad.json"
    path.write_text('{"num_qubits": 2, "ops": [{"gate": "FOO", "col": 0, "qubits": [0]}]}')
    with pytest.raises(ValueError, match="unknown gate"):
        Composer(file=path)
    path.write_text("not json")
    with pytest.raises(ValueError, match="not a JSON file"):
        Composer(file=path)


def test_save_crops_unused_last_qubits(tmp_path):
    pytest.importorskip("anywidget")
    import json  # pylint: disable=import-outside-toplevel
    from ket.composer import Composer  # pylint: disable=import-outside-toplevel

    path = tmp_path / "crop.json"
    comp = Composer(num_qubits=5)
    comp.ops = [
        {"gate": "H", "col": 0, "qubits": [0], "params": []},
        {"gate": "CTRL", "col": 1, "qubits": [0], "params": []},
        {"gate": "X", "col": 1, "qubits": [2], "params": []},
    ]
    comp.save(path)
    assert comp.num_qubits == 5  # The editor keeps its qubits
    assert json.loads(path.read_text())["num_qubits"] == 3  # q1 kept, q3 and q4 cropped
    assert Composer(file=path).num_qubits == 3

    comp.clear()
    comp.save(path)
    assert json.loads(path.read_text())["num_qubits"] == 1


def _same_up_to_phase(a, b):
    overlap = sum(a.get(k, 0).conjugate() * v for k, v in b.items())
    return math.isclose(abs(overlap), 1.0, abs_tol=1e-5)


def _mixed(q):
    H(q[:3])
    T(q[0])
    adj(S)(q[1])
    RX(0.3, q[2])
    with control(q[0] == 0):
        RY(math.pi / 3, q[3])
    ctrl(q[:2], Z)(q[2])
    SWAP(q[1], q[3])
    P(1.25, q[0])
    adj(RZ(0.7))(q[3])


@pytest.mark.parametrize(
    "gate, size",
    [
        (lambda q: CNOT(H(q[0]), q[1]), 2),
        (QFT, 3),
        (QFT, 4),
        (_mixed, 4),
    ],
)
def test_circuit_from_gate_matches(gate, size):
    data = circuit_from_gate(gate, size)
    Circuit(data)  # A valid composer circuit

    p = Process()
    q = p.alloc(size)
    H(q)  # Not only |0...0>: compare the full action of the gate
    RY(0.4, q[0])
    gate(q)
    reference = dump(q).states

    p = Process()
    q = p.alloc(size)
    H(q)
    RY(0.4, q[0])
    circuit_gate(data)(q)
    assert _same_up_to_phase(reference, dump(q).states)


def test_circuit_from_gate_layout():
    def gate(q):
        H(q[0])
        H(q[1])  # Same column as H(q[0])
        CNOT(q[0], q[1])  # Own column, with a control
        X(q[2])  # Not into the controlled column: next free one
        S(q[0])
        RX(math.pi / 2, q[1])

    ops = circuit_from_gate(gate, 3)["ops"]
    by_col = {}
    for op in ops:
        by_col.setdefault(op["col"], []).append((op["gate"], op["qubits"], op["params"]))
    assert by_col[0] == [("H", [0], []), ("H", [1], []), ("X", [2], [])]
    assert by_col[1] == [("CTRL", [0], []), ("X", [1], [])]
    assert by_col[2] == [("P", [0], ["pi/2"]), ("RX", [1], ["pi/2"])]


def test_circuit_from_gate_keeps_phases():
    # One phase per qubit: Ket merges consecutive phases on the same qubit.
    def gate(q):
        S(q[0])
        T(q[1])
        adj(T)(q[2])
        P(1.25, q[3])

    ops = circuit_from_gate(gate, 4)["ops"]
    assert [(op["gate"], op["params"]) for op in ops] == [
        ("P", ["pi/2"]),
        ("P", ["pi/4"]),
        ("P", ["-pi/4"]),
        ("P", ["1.25"]),
    ]


def test_circuit_from_gate_aux_qubits():
    def gate(q):
        aux = q.ket_process.alloc(1)
        CNOT(q[0], aux)

    with pytest.raises(ValueError, match="auxiliary"):
        circuit_from_gate(gate, 2)


def test_widget_from_gate(tmp_path):
    pytest.importorskip("anywidget")
    from ket.composer import Composer  # pylint: disable=import-outside-toplevel

    comp = Composer(num_qubits=3, gate=QFT)
    assert comp.error == ""
    assert comp.ops
    p = Process()
    q = p.alloc(3)
    X(q[2])
    comp(q)
    states = dump(q).states
    p = Process()
    q = p.alloc(3)
    X(q[2])
    QFT(q)
    assert _same_up_to_phase(dump(q).states, states)

    with pytest.raises(ValueError, match="not both"):
        Composer(num_qubits=3, gate=QFT, file=tmp_path / "x.json")


@pytest.mark.parametrize(
    "prepare, expected",
    [
        (lambda q: None, [[0, 0, 1]]),
        (X, [[0, 0, -1]]),
        (H, [[1, 0, 0]]),
        (lambda q: S(H(q)), [[0, 1, 0]]),
        (lambda q: RY(math.pi / 3, q), [[math.sin(math.pi / 3), 0, 0.5]]),
        # Entangled: both vectors at the center of the sphere.
        (lambda q: CNOT(H(q[0]), q[1]), [[0, 0, 0], [0, 0, 0]]),
        # Product state: each qubit keeps its own vector.
        (lambda q: (S(H(q[0])), X(q[1])), [[0, 1, 0], [0, 0, -1]]),
    ],
)
def test_bloch_vectors(prepare, expected):
    from ket.composer import bloch_vectors  # pylint: disable=import-outside-toplevel

    p = Process()
    q = p.alloc(len(expected))
    prepare(q)
    vectors = bloch_vectors(dump(q).states, len(expected))
    for got, want in zip(vectors, expected):
        assert all(math.isclose(a, b, abs_tol=1e-5) for a, b in zip(got, want))


def test_simulation_has_bloch_vectors():
    result = simulate_circuit(GHZ)
    assert len(result["bloch"]) == 3
    assert all(abs(v) < 1e-6 for vec in result["bloch"] for v in vec)


def test_cnot_gate():
    data = {
        "num_qubits": 3,
        "ops": [
            {"gate": "H", "col": 0, "qubits": [1]},
            {"gate": "CNOT", "col": 1, "qubits": [1, 0]},  # control q1, target q0
            {"gate": "CTRL", "col": 2, "qubits": [0]},
            {"gate": "CNOT", "col": 2, "qubits": [1, 2]},  # Toffoli
        ],
    }
    p = Process()
    q = circuit_gate(data)(p.alloc(3))
    assert set(dump(q).states) == {0, 0b111}
    assert "CNOT(q[1], q[0])" in circuit_code(data)
    assert "ctrl(q[0], CNOT)(q[1], q[2])" in circuit_code(data)


def test_code_uses_ctrl():
    code = circuit_code(PARAM)
    assert "with control" not in code
    assert "ctrl(q[0], X)(q[1])" in code
    assert "ctrl(q[1] == 0b0, RY(2*theta + pi/8))(q[2])" in code
    assert "from ket import H, RY, SWAP, X, ctrl" in code


REGS = {
    "num_qubits": 3,
    "registers": [["a", 2], ["b", 1]],
    "ops": [
        {"gate": "H", "col": 0, "qubits": [0]},
        {"gate": "CNOT", "col": 1, "qubits": [0, 2]},
        {"gate": "ACTRL", "col": 2, "qubits": [1]},
        {"gate": "CTRL", "col": 2, "qubits": [2]},
        {"gate": "X", "col": 2, "qubits": [0]},
    ],
}


def test_registers_gate_and_code():
    gate = circuit_gate(REGS)
    p = Process()
    a, b = p.alloc(2), p.alloc(1)
    assert gate(a, b) == (a, b)
    assert set(dump(a + b).states) == {0b000, 0b001}

    with pytest.raises(TypeError, match=r"takes 2 qubit register\(s\) \(a, b\), got 1"):
        gate(a + b)
    with pytest.raises(ValueError, match="register 'b' takes 1 qubit"):
        gate(a, p.alloc(2))

    code = circuit_code(REGS)
    assert "def circuit(a, b):" in code
    assert "CNOT(a[0], b[0])" in code
    assert "ctrl((a[1] + b[0]) == 0b01, X)(a[0])" in code
    assert "return a, b" in code
    namespace = {}
    exec(code, namespace)  # pylint: disable=exec-used
    p = Process()
    a, b = p.alloc(2), p.alloc(1)
    namespace["circuit"](a, b)
    assert set(dump(a + b).states) == {0b000, 0b001}


@pytest.mark.parametrize(
    "registers, message",
    [
        ([["a", 1], ["a", 1]], "two registers"),
        ([["X", 1]], "reserved"),
        ([["pi", 1]], "reserved"),
        ([["1a", 1]], "not a valid register name"),
        ([["a", 0]], "positive"),
    ],
)
def test_invalid_registers(registers, message):
    size = sum(max(n, 0) for _, n in registers) or 1
    with pytest.raises(ValueError, match=message):
        Circuit({"num_qubits": size, "registers": registers, "ops": []})


def test_register_and_parameter_name_clash():
    data = {
        "num_qubits": 1,
        "registers": [["theta", 1]],
        "ops": [{"gate": "RX", "col": 0, "qubits": [0], "params": ["theta"]}],
    }
    with pytest.raises(ValueError, match="both a register and an angle parameter"):
        Circuit(data)


def test_circuit_from_gate_uses_parameter_names():
    def oracle(inputs, target):
        ctrl(inputs, X)(target)

    data = circuit_from_gate(oracle, [2, 1])
    assert data["registers"] == [["inputs", 2], ["target", 1]]
    assert circuit_from_gate(oracle, {"x": 2, "y": 1})["registers"] == [["x", 2], ["y", 1]]
    assert circuit_from_gate(QFT, 3)["registers"] == [["qubits", 3]]
    assert circuit_from_gate(lambda q: H(q), 2)["registers"] == [["q", 2]]

    gate = circuit_gate(data)
    p = Process()
    inputs, target = p.alloc(2), p.alloc(1)
    X(inputs)
    gate(inputs, target)
    assert set(dump(inputs + target).states) == {0b111}


def test_widget_registers(tmp_path):
    pytest.importorskip("anywidget")
    from ket.composer import Composer  # pylint: disable=import-outside-toplevel

    comp = Composer(num_qubits={"a": 2, "b": 1})
    assert comp.registers == [["a", 2], ["b", 1]] and comp.num_qubits == 3
    comp.ops = list(REGS["ops"])
    assert comp.error == ""
    p = Process()
    comp(p.alloc(2), p.alloc(1))
    assert "def circuit(a, b):" in comp.to_code()

    # Adding and removing qubits changes the last register.
    comp.num_qubits = 4
    assert comp.registers == [["a", 2], ["b", 2]]
    comp.num_qubits = 2
    assert comp.registers == [["a", 2]]

    # With gate=, the registers are named after the gate parameters.
    def oracle(inputs, target):
        ctrl(inputs, X)(target)

    comp = Composer(num_qubits=[2, 1], gate=oracle)
    assert comp.registers == [["inputs", 2], ["target", 1]]

    # Saving crops each register after its last used qubit.
    comp = Composer(num_qubits={"a": 3, "b": 2})
    comp.ops = [
        {"gate": "H", "col": 0, "qubits": [1], "params": []},
        {"gate": "X", "col": 0, "qubits": [3], "params": []},
    ]
    path = tmp_path / "regs.json"
    comp.save(path)
    opened = Composer(file=path)
    assert opened.registers == [["a", 2], ["b", 1]]
    assert sorted(op["qubits"] for op in opened.ops) == [[1], [2]]


GROUPED = {
    "num_qubits": 4,
    "ops": [
        {"gate": "H", "col": 0, "qubits": [0], "params": [], "group": "g1"},
        {"gate": "CNOT", "col": 1, "qubits": [0, 1], "params": [], "group": "g1"},
        {"gate": "X", "col": 0, "qubits": [2], "params": []},
    ],
}


def test_groups_do_not_change_the_gate():
    plain = {**GROUPED, "ops": [{k: v for k, v in op.items() if k != "group"} for op in GROUPED["ops"]]}
    p = Process()
    a = circuit_gate(GROUPED)(p.alloc(4))
    p = Process()
    b = circuit_gate(plain)(p.alloc(4))
    assert _close(dump(a).states, dump(b).states)
    assert circuit_code(GROUPED) == circuit_code(plain)


def test_groups_survive_crop_and_save(tmp_path):
    cropped = Circuit(GROUPED).cropped()
    assert cropped["num_qubits"] == 3
    assert sorted((op["gate"], op.get("group")) for op in cropped["ops"]) == [
        ("CNOT", "g1"),
        ("H", "g1"),
        ("X", None),
    ]
    with pytest.raises(ValueError, match="group must be a string"):
        Circuit({"num_qubits": 1, "ops": [{"gate": "X", "col": 0, "qubits": [0], "group": 3}]})

    pytest.importorskip("anywidget")
    from ket.composer import Composer  # pylint: disable=import-outside-toplevel

    comp = Composer(num_qubits=4)
    comp.ops = list(GROUPED["ops"])
    comp.save(tmp_path / "grouped.json")
    opened = Composer(file=tmp_path / "grouped.json")
    assert sorted(op.get("group") or "" for op in opened.ops) == ["", "g1", "g1"]


def test_greek_parameter_names():
    data = {
        "num_qubits": 1,
        "params": {"alpha": 0.2, "θ": 0.3},
        "ops": [
            {"gate": "RX", "col": 0, "qubits": [0], "params": ["2*alpha + θ"]},
            {"gate": "RZ", "col": 1, "qubits": [0], "params": ["lambda/2"]},  # keyword -> lambda_
            {"gate": "RY", "col": 2, "qubits": [0], "params": ["Omega + theta_1"]},
        ],
    }
    circuit = Circuit(data)
    assert circuit.param_names == ["alpha", "θ", "lambda_", "Omega", "theta_1"]

    # Parameters without defaults may follow those with defaults: keyword-only.
    code = circuit_code(data)
    assert "def circuit(q, *, alpha=0.2, θ=0.3, lambda_, Omega, theta_1):" in code
    namespace = {}
    exec(code, namespace)  # pylint: disable=exec-used

    def reference(q):
        RX(2 * 0.2 + 0.3, q)
        RZ(0.5 / 2, q)
        RY(0.1 + 0.4, q)

    p = Process()
    q = p.alloc(1)
    namespace["circuit"](q, lambda_=0.5, Omega=0.1, theta_1=0.4)
    from_code = dump(q).states
    p = Process()
    q = p.alloc(1)
    circuit_gate(data)(q, lambda_=0.5, Omega=0.1, theta_1=0.4)
    from_gate = dump(q).states
    p = Process()
    q = p.alloc(1)
    reference(q)
    assert _close(from_code, dump(q).states) and _close(from_gate, dump(q).states)
