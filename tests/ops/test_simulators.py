# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

"""exp_value and sample agree across the KBW simulators."""

import random

import pytest

from ket import CNOT, H, Process, RX, RY, RZ, X, Y, Z, exp_value, obs, sample


def _available(simulator, num_qubits):
    try:
        Process(simulator=simulator, num_qubits=num_qubits, execution="live")
        return True
    except Exception:  # pylint: disable=broad-except
        return False


SIMULATORS = [
    pytest.param("dense", id="dense"),
    pytest.param("sparse", id="sparse"),
    pytest.param("sparse v1", id="sparse v1"),
    pytest.param(
        "dense gpu",
        id="dense gpu",
        marks=pytest.mark.skipif(not _available("dense gpu", 2), reason="no GPU simulator"),
    ),
    pytest.param(
        "sparse gpu",
        id="sparse gpu",
        marks=pytest.mark.skipif(not _available("sparse gpu", 2), reason="no GPU simulator"),
    ),
]


def _circuit(q, n, seed):
    rng = random.Random(seed)
    for i in range(n):
        H(q[i])
        RX(rng.uniform(0, 3), q[i])
        RZ(rng.uniform(0, 3), q[i])
    for i in range(n - 1):
        CNOT(q[i], q[i + 1])
    for i in range(n):
        RY(rng.uniform(0, 3), q[i])


def _hamiltonian(q, n, seed):
    rng = random.Random(seed)
    with obs():
        h = 0.5
        for _ in range(40):
            factors = [rng.choice((X, Y, Z))(q[j]) for j in rng.sample(range(n), rng.randint(1, 3))]
            term = factors[0]
            for factor in factors[1:]:
                term = term * factor
            h = h + rng.uniform(-1, 1) * term
        return h


def _exp_value(simulator, n, seed):
    p = Process(simulator=simulator, num_qubits=n, execution="live")
    q = p.alloc(n)
    _circuit(q, n, seed)
    return exp_value(_hamiltonian(q, n, seed + 1)).get()


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_exp_value_matches_dense(simulator):
    n = 8
    assert _exp_value(simulator, n, 3) == pytest.approx(_exp_value("dense", n, 3), abs=1e-5)


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_exp_value_does_not_change_state(simulator):
    n = 6
    p = Process(simulator=simulator, num_qubits=n, execution="live")
    q = p.alloc(n)
    _circuit(q, n, 5)
    h = _hamiltonian(q, n, 6)
    first = exp_value(h).get()
    assert exp_value(h).get() == pytest.approx(first, abs=1e-6)


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_sample_bit_order(simulator):
    p = Process(simulator=simulator, num_qubits=4, execution="live")
    q = p.alloc(4)
    X(q[0])
    X(q[2])  # |1010>, q[0] is the most significant bit
    assert sample(q, 100).get() == {0b1010: 100}
    assert sample(q[2] + q[1] + q[0], 50).get() == {0b101: 50}
    assert sample(q[3], 10).get() == {0: 10}


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_sample_distribution(simulator):
    p = Process(simulator=simulator, num_qubits=3, execution="live")
    q = p.alloc(3)
    RY(2 * 0.5, q[0])  # P(1) = sin(0.5)^2 ~ 0.2298
    CNOT(q[0], q[2])
    shots = 40_000
    counts = sample(q, shots).get()
    assert set(counts) <= {0b000, 0b101}
    assert sum(counts.values()) == shots
    assert counts.get(0b101, 0) / shots == pytest.approx(0.2298, abs=0.01)


@pytest.mark.skipif(not _available("dense gpu", 2), reason="no GPU simulator")
def test_gpu_block_path(monkeypatch):
    # Force blocks of 20 qubits (no tuning): 21 qubits uses the block
    # simulator, and gates on high qubits permute them.
    monkeypatch.setenv("KBW_BLOCK_SIZE", "20")
    n = 21
    p = Process(simulator="dense gpu", num_qubits=n, execution="live")
    q = p.alloc(n)
    X(q[20])
    X(q[0])
    H(q[19])
    CNOT(q[19], q[1])
    counts = sample(q[0] + q[1] + q[19] + q[20], 20_000).get()
    assert set(counts) == {0b1001, 0b1111}
    with obs():
        h = Z(q[0]) + 2 * Z(q[20]) + 3 * X(q[19]) * X(q[1]) - Y(q[5])
    assert exp_value(h).get() == pytest.approx(-1 - 2 + 3, abs=1e-5)


@pytest.mark.skipif(not _available("dense gpu", 2), reason="no GPU simulator")
def test_gpu_block_size_tuning_is_stored(tmp_path, monkeypatch):
    import json
    import time

    from ket.clib.kbw import NO_BLOCKING, tune_gpu_block_size

    monkeypatch.setenv("KBW_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("KBW_BLOCK_SIZE", raising=False)

    size = tune_gpu_block_size(force=True)
    assert size == NO_BLOCKING or 12 <= size < 22

    stored = json.loads((tmp_path / "gpu_tuning.json").read_text())
    (entry,) = stored["devices"].values()
    assert entry["block_size"] == size
    assert entry["seconds"]

    start = time.perf_counter()
    assert tune_gpu_block_size() == size  # read back, no measurement
    assert time.perf_counter() - start < 0.5
