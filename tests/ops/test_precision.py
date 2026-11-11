# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Single and double precision simulators."""

import logging
from math import cos

import pytest

from ket import RX, Z, CNOT, H, Process, exp_value, obs, sample
from ket.clib.kbw import get_simulator


def _available(simulator):
    try:
        Process(simulator=simulator, num_qubits=2)
        return True
    except Exception:  # pylint: disable=broad-except
        return False


def _error(simulator, precision):
    """Error of ⟨Z⟩ after 1000 small rotations, which add up to RX(1)."""
    p = Process(simulator=simulator, num_qubits=2, precision=precision)
    q = p.alloc()
    for _ in range(1000):
        RX(1e-3, q)
    with obs():
        h = Z(q)
    return abs(exp_value(h).get() - cos(1.0))


@pytest.mark.parametrize("simulator", ["dense", "dense v1", "sparse", "sparse v1"])
def test_double_precision_is_more_accurate(simulator):
    assert _error(simulator, "single") > 1e-9
    assert _error(simulator, "double") < 1e-13


@pytest.mark.parametrize("simulator", ["dense", "sparse"])
def test_double_precision_runs_circuits(simulator):
    p = Process(simulator=simulator, num_qubits=3, precision="double", execution="batch")
    q = p.alloc(3)
    CNOT(H(q[0]), q[1])
    CNOT(q[1], q[2])
    assert set(sample(q).get()) == {0, 7}


def test_double_precision_with_gradient():
    p = Process(gradient=True, precision="double")
    theta = p.param(0.5)
    q = p.alloc()
    RX(theta, q)
    with obs():
        h = Z(q)
    exp_value(h).get()
    assert theta.grad == pytest.approx(-0.479425538604203, abs=1e-12)


@pytest.mark.parametrize("simulator", ["dense gpu", "sparse gpu"])
def test_gpu_simulators_warn_and_use_single_precision(simulator, caplog):
    if not _available(simulator):
        pytest.skip("no GPU simulator")
    with caplog.at_level(logging.WARNING, logger="ket"):
        error = _error(simulator, "double")
    assert error > 1e-9
    assert any("only supports single precision" in r.getMessage() for r in caplog.records)


def test_single_precision_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="ket"):
        Process(simulator="dense", num_qubits=2)
    assert not any("precision" in r.getMessage() for r in caplog.records)


def test_invalid_precision():
    with pytest.raises(ValueError, match="unknown precision"):
        Process(precision="half")
    with pytest.raises(ValueError, match="unknown precision"):
        get_simulator(2, precision="quad")


def test_precision_needs_kbw():
    class _Target:  # pylint: disable=too-few-public-methods
        pass

    with pytest.raises(ValueError, match="Cannot specify arguments"):
        Process(_Target(), precision="double")
