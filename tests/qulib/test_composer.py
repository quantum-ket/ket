# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

import cmath
import math

import pytest

from ket import Process, dump, H, RY, SWAP, X, ctrl, control
from ket.qulib.composer import circuit_gate, circuit_code, simulate_circuit

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
    assert "def circuit(q, theta=0.4):" in code
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
    with pytest.raises(ValueError, match="acts on 3 qubit"):
        gate(p.alloc(2), theta=0.1)
    with pytest.raises(ValueError, match="acts on 3 qubit"):
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
    with pytest.raises(ValueError, match="acts on 3 to 5 qubit"):
        gate(p.alloc(2))
    with pytest.raises(ValueError, match="acts on 3 to 5 qubit"):
        gate(p.alloc(6))


def test_widget():
    pytest.importorskip("anywidget")
    from ket.qulib.composer import Composer  # pylint: disable=import-outside-toplevel

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


def test_qulib_composer_shortcut():
    pytest.importorskip("anywidget")
    from ket import qulib  # pylint: disable=import-outside-toplevel
    from ket.qulib.composer import Composer  # pylint: disable=import-outside-toplevel

    comp = qulib.composer(num_qubits=4)
    assert isinstance(comp, Composer)
    assert comp.num_qubits == 4


def test_save_and_load(tmp_path):
    pytest.importorskip("anywidget")
    from ket.qulib.composer import Composer  # pylint: disable=import-outside-toplevel

    path = tmp_path / "param.json"
    comp = Composer(num_qubits=3, file=path)
    assert comp.ops == [] and comp.file == str(path)  # No file yet: empty
    comp.params = dict(PARAM["params"])
    comp.ops = list(PARAM["ops"])
    comp.save()
    assert path.exists()

    opened = Composer(file=path)  # num_qubits comes from the file
    assert opened.num_qubits == 3
    assert opened.to_dict() == comp.to_dict()
    assert _close(_states(opened), _states(_reference, 0.4))

    other = Composer(num_qubits=1)
    other.load(path)
    assert other.to_dict() == comp.to_dict()


def test_save_and_load_buttons(tmp_path):
    pytest.importorskip("anywidget")
    from ket.qulib.composer import Composer  # pylint: disable=import-outside-toplevel

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
    from ket.qulib.composer import Composer  # pylint: disable=import-outside-toplevel

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
    from ket.qulib.composer import Composer  # pylint: disable=import-outside-toplevel

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
