"""Verification helpers for the compute-action-uncompute pattern.

Used by :func:`~ket.operations.around` and :func:`~ket.operations.undo` to
check the safe use of auxiliary qubits (clean auxiliary qubit pattern) and to
decide when the approximated decomposition can be enabled. They query the
per-qubit properties (gate classification and read qubits) that Libket
records for each block.
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access

from ctypes import c_size_t
import json
from typing import TYPE_CHECKING

from ket.clib.libket import API as libket

if TYPE_CHECKING:
    from .base import Process


def _block_proprieties(block) -> dict[int, dict]:
    """Return the per-qubit properties of a block, keyed by qubit index."""
    proprieties_ptr = block.proprieties_json()
    proprieties = json.loads(proprieties_ptr.value)
    libket["ket_string_delete"](proprieties_ptr)
    return {int(target): op for target, op in proprieties.items()}


def _block_proprieties_subset(block, qubits) -> dict[int, dict]:
    """Return the per-qubit properties of a block for the given qubits only.

    Equivalent to ``{q: p for q, p in _block_proprieties(block).items() if q in qubits}``,
    but Libket only serializes the requested qubits (using its cached records),
    so the cost does not grow with the size of the block.
    """
    qubits = list(qubits)
    array = (c_size_t * len(qubits))(*qubits)
    proprieties_ptr = block.proprieties_subset_json(array, len(qubits))
    proprieties = json.loads(proprieties_ptr.value)
    libket["ket_string_delete"](proprieties_ptr)
    return {int(target): op for target, op in proprieties.items()}


def _without_control_aux(
    proprieties: dict[int, dict], control_aux: set[int]
) -> dict[int, dict]:
    """Remove the shared-control auxiliary qubits from block properties.

    A shared-control auxiliary qubit (see ``ket.operations.control``) holds the
    AND of a set of control qubits only between its compute and uncompute
    steps, which are always appended together, so it is |0> outside them. It
    is removed from the properties, and a qubit that reads it reads, instead,
    the qubits it was computed from (transitively).
    """

    def sources(qubit, seen):
        result = set()
        for read in proprieties.get(qubit, {}).get("read_qubits", ()):
            if read in control_aux:
                if read not in seen:
                    seen.add(read)
                    result |= sources(read, seen)
            else:
                result.add(read)
        return result

    filtered = {}
    for target, op in proprieties.items():
        if target in control_aux:
            continue
        read_qubits = set(op["read_qubits"])
        if read_qubits & control_aux:
            read_qubits = (read_qubits - control_aux) | sources(target, set())
            op = {**op, "read_qubits": sorted(read_qubits)}
        filtered[target] = op
    return filtered


def _analyze_compute(proprieties: dict[int, dict], ket_process: Process):
    """Check the compute block of a compute-action-uncompute pattern.

    Returns the set of qubits that must remain unchanged (except for diagonal
    operations) until the uncomputation, and whether every qubit modified by
    the compute block is classified as identity, diagonal, or permutation.

    The returned blocked set is the transitive closure of the qubits read by
    the auxiliary qubits: if an auxiliary qubit reads ``q`` and ``q`` reads
    ``r`` in the compute block, both ``q`` and ``r`` are blocked.

    Raises:
        RuntimeError: If an auxiliary qubit is subject to a non-permutation
            operation, or if a qubit read (directly or transitively) by an
            auxiliary qubit is subject to a non-permutation operation in the
            compute block.
    """

    if ket_process._control_aux:
        proprieties = _without_control_aux(proprieties, ket_process._control_aux)

    blocked_qubits = set()
    for target, op in proprieties.items():
        if not ket_process._is_aux(target):
            continue
        match op["propriety"]:
            case "Permutation":
                blocked_qubits.update(op["read_qubits"])
            case "Unitary":
                raise RuntimeError("Operation not allowed on axillary qubit.")

    pending = list(blocked_qubits)
    while pending:
        qubit = pending.pop()
        op = proprieties.get(qubit)
        if op is None:
            continue
        if op["propriety"] == "Unitary":
            raise RuntimeError(
                "Operation violates uncomputation: a qubit read by an auxiliary "
                "qubit is subject to a non-permutation operation in the compute block."
            )
        for read_qubit in op["read_qubits"]:
            if read_qubit not in blocked_qubits:
                blocked_qubits.add(read_qubit)
                pending.append(read_qubit)

    is_permutation = all(
        op["propriety"] in ("Identity", "Diagonal", "Permutation")
        for op in proprieties.values()
    )

    return blocked_qubits, is_permutation


def _touched_qubits(proprieties: dict[int, dict]) -> set[int]:
    """Return the qubits a block acts on, either as target or as control."""
    touched = set(proprieties)
    for op in proprieties.values():
        touched.update(op["read_qubits"])
    return touched
