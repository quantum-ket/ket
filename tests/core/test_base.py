# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from ket.base import Process, Quant


def test_process_initialization():
    """Test process initialization with different configurations."""
    # Test sparse simulator initialization
    p_sparse = Process(simulator="sparse")
    assert p_sparse is not None

    # Test dense simulator with specific qubit count
    p_dense = Process(simulator="dense", num_qubits=5)
    assert p_dense is not None

    # Test batch execution mode
    p_batch = Process(execution="batch")
    assert p_batch is not None


def test_process_conflicting_args():
    """Test that Process fails when providing both a target and specific configs."""

    class DummyTarget:
        def connect(self):
            return None

    # Should raise ValueError if execution_target is mixed with simulator/num_qubits
    with pytest.raises(
        ValueError, match="Cannot specify arguments if configuration is provided"
    ):
        Process(execution_target=DummyTarget(), simulator="sparse")


def test_quant_allocation():
    """Test qubit allocation and expected failures."""
    p = Process()

    # Happy path: allocate 3 qubits
    q = p.alloc(3)
    assert len(q) == 3
    assert q.qubits == [0, 1, 2]

    # Expected failure: allocate less than 1 qubit
    with pytest.raises(ValueError, match="Cannot allocate less than 1 qubit"):
        p.alloc(0)
    with pytest.raises(ValueError, match="Cannot allocate less than 1 qubit"):
        p.alloc(-5)


def test_aux_allocation_and_free():
    """Test the lifecycle of auxiliary qubits."""
    p = Process()

    # Allocate 2 auxiliary qubits
    aux = p.alloc_aux(2)
    aux_qubits = aux.qubits.copy()
    assert len(aux) == 2
    assert p._is_aux(aux_qubits[0]) is True

    # To test manual free without triggering the GC warning,
    # we use the context manager or call the finalizer directly.
    # In this case, we call the finalizer which executes _free_aux and
    # ensures it won't run again when the object is deleted.
    aux._finalizer()

    assert aux_qubits[0] in p._qubit_pool
    assert p._is_aux(aux_qubits[0]) is False

    # Expected failure: trying to free a non-auxiliary qubit
    q = p.alloc(1)
    with pytest.raises(RuntimeError, match="Cannot free an non-auxiliary qubit"):
        p._free_aux(q.qubits)


def test_quant_concatenation():
    """Test Quant addition and its safety constraints."""
    p = Process()
    q1 = p.alloc(2)
    q2 = p.alloc(2)

    # Success path
    q3 = q1 + q2
    assert len(q3) == 4
    assert q3.qubits == q1.qubits + q2.qubits

    # Failure: overlapping indices (same register)
    with pytest.raises(
        ValueError, match="Cannot concatenate qubits with overlapping indices"
    ):
        _ = q1 + q1

    # Failure: qubits from different Ket processes
    p2 = Process()
    q_other = p2.alloc(2)
    with pytest.raises(
        ValueError, match="Cannot concatenate qubits from different processes"
    ):
        _ = q1 + q_other


def test_quant_slicing_and_indexing():
    """Test native Python list-like operations on Quant objects."""
    p = Process()
    q = p.alloc(5)

    # Basic Indexing
    assert q[0].qubits == [q.qubits[0]]

    # Slicing
    sliced = q[1:4]
    assert len(sliced) == 3
    assert sliced.qubits == q.qubits[1:4]

    # Reversed operation
    rev = reversed(q)
    assert rev.qubits == list(reversed(q.qubits))

    # At method for arbitrary index selection
    subset = q.at([0, 3])
    assert len(subset) == 2
    assert subset.qubits == [q.qubits[0], q.qubits[3]]
