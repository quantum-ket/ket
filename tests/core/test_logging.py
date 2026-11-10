# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Logging, error messages, and the C callback bridge."""

import gc
import logging

import pytest

import ket
from ket import CNOT, H, X, Process, control, measure, sample
from ket.clib.libket.execution import BatchExecution
from ket.clib.log import _logger_name, _to_rust
from ket.clib.wrapper import LibketError


@pytest.fixture
def debug_logs(caplog):
    """Capture records of the `ket` loggers at debug level."""
    ket.set_log_level("debug")
    caplog.set_level(logging.DEBUG, logger="ket")
    yield caplog
    ket.set_log_level(logging.NOTSET)


def test_rust_records_reach_python_loggers(debug_logs):
    p = Process(num_qubits=2, simulator="dense", execution="batch")
    q = p.alloc(2)
    CNOT(H(q[0]), q[1])
    sample(q).get()

    names = {record.name for record in debug_logs.records}
    assert "ket.base" in names
    assert "ket.kbw.c_api" in names
    assert "ket.libket.process" in names
    assert any("executing batch" in r.getMessage() for r in debug_logs.records)


def test_records_below_level_are_not_sent(caplog):
    ket.set_log_level("warning")
    caplog.set_level(logging.DEBUG)  # the root logger, not `ket`
    Process(num_qubits=2)
    assert not [r for r in caplog.records if r.name.startswith("ket.libket")]
    ket.set_log_level(logging.NOTSET)


def test_invalid_kbw_seed_is_a_warning(monkeypatch, caplog):
    monkeypatch.setenv("KBW_SEED", "not a number")
    with caplog.at_level(logging.WARNING, logger="ket"):
        Process(num_qubits=2)
    assert any("KBW_SEED" in r.getMessage() for r in caplog.records)


def test_log_level_mapping():
    assert _to_rust(logging.WARNING) == 2
    assert _to_rust(logging.DEBUG) == 4
    assert _to_rust(5) == 5
    assert _to_rust(logging.CRITICAL + 10) == 0
    assert _logger_name("ket.libket", "ket::process") == "ket.libket.process"
    assert _logger_name("ket.kbw", "wgpu_core::device") == "ket.kbw.wgpu_core.device"
    with pytest.raises(ValueError):
        ket.set_log_level("loud")


def _logged_errors(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


def test_error_cause_is_logged(caplog):
    with caplog.at_level(logging.ERROR, logger="ket"):
        with pytest.raises(LibketError, match="^KBW: "):
            Process(num_qubits=2, simulator="nope")
        with pytest.raises(LibketError, match="^KBW: "):
            Process(num_qubits=40, simulator="dense")
    errors = _logged_errors(caplog)
    assert any('unknown simulator "nope"' in e for e in errors)
    assert any("supports up to 32 qubits, got 40" in e for e in errors)


def test_short_error_message_and_cause_in_log(caplog):
    p = Process(num_qubits=2)
    q = p.alloc(2)
    with caplog.at_level(logging.ERROR, logger="ket"):
        with pytest.raises(
            LibketError,
            match="^Libket: a qubit cannot be both a control and the target of the same gate$",
        ):
            with control(q[0]):
                X(q[0])
    assert any("qubit 0 controls its own PauliX gate" in e for e in _logged_errors(caplog))


@pytest.mark.parametrize(
    "source, cause",
    [
        ("qreg q[2];\nh q[0]\ncx q[0], q[1];", "unexpected `cx` at line 5, column 1"),
        ("qreg q[2];\nfoo q[0];", "undefined gate `foo`"),
        ("qreg q[2];\nU(0, 0, 0) q[5];", "`q[5]` is out of range: `q` has 2 qubits"),
    ],
)
def test_openqasm_error_cause_is_logged(caplog, source, cause):
    p = Process(num_qubits=4)
    q = p.alloc(2)
    with caplog.at_level(logging.ERROR, logger="ket"):
        with pytest.raises(LibketError, match="OpenQASM"):
            ket.qasm(q, 'OPENQASM 2.0;\ninclude "qelib1.inc";\n' + source)
    assert any(cause in e for e in _logged_errors(caplog))


@pytest.mark.parametrize(
    "graph, cause",
    [
        ([], "no path from qubit 0 to qubit 1"),
        ([(0, 1), (2, 3)], "no path from qubit 0 to qubit 2"),
        ([(0, 1), (1, 2), (2, 9)], "edge (2, 9)"),
    ],
)
def test_invalid_coupling_graph_is_an_error(caplog, graph, cause):
    p = Process(
        num_qubits=4, simulator="dense", execution="batch", coupling_graph=graph
    )
    q = p.alloc(4)
    H(q)
    CNOT(q[0], q[3])
    with caplog.at_level(logging.ERROR, logger="ket"):
        with pytest.raises(LibketError, match="invalid coupling graph"):
            sample(q).get()
    assert any(cause in e for e in _logged_errors(caplog))


class _Flaky(BatchExecution):
    """Batch target whose first `sample` call fails."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def sample(self, gates, qubits_to_sample, shots):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("network down")
        return [[[0], [3]], [shots // 2, shots - shots // 2]]

    def connect(self):
        return self.configure(2)


def test_callback_exception_reaches_the_caller_and_retry_works():
    p = Process(_Flaky())
    q = p.alloc(2)
    CNOT(H(q[0]), q[1])
    result = sample(q)
    with pytest.raises(ConnectionError, match="network down"):
        result.get()
    # The failed execution must leave the process able to run again.
    assert result.get() == {0: 1024, 3: 1024}


def test_block_set_does_not_free_twice():
    p = Process(num_qubits=3)
    q = p.alloc(3)
    for _ in range(200):
        with p.block_builder(inverse=True, control=[q.qubits[2]], append=False):
            X(q[0])
            CNOT(q[0], q[1])
        gc.collect()


@pytest.mark.parametrize("simulator", ["dense", "dense v1", "sparse", "sparse v1"])
def test_live_decompose_keeps_state_between_gates(simulator):
    p = Process(num_qubits=3, simulator=simulator, decompose=True)
    a, b, c = p.alloc(3)
    X(a)
    assert measure(a).value == 1
    CNOT(a, b)
    with control(a + b):
        X(c)
    assert measure(b + c).value == 0b11


def test_decomposition_and_shared_control_are_logged(caplog):
    with caplog.at_level(logging.DEBUG, logger="ket"):
        ket.set_log_level("debug")
        p = Process(simulator="dense", num_qubits=12, decompose=True)
        q = p.alloc(8)
        H(q[:5])
        with control(q[:5]):
            X(q[6])
            H(q[7])
        measure(q)
    ket.set_log_level(logging.NOTSET)
    info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any(m.startswith("sharing 5 controls [0, 1, 2, 3, 4]") for m in info)
    assert any(
        m.startswith("decomposed ") and "Network C3X" in m and "Single control" in m
        for m in debug
    )
