"""Shared multi-qubit controls through a clean auxiliary qubit.

Used by :func:`~ket.operations.control` when a block is controlled by two or
more qubits.
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access

from typing import TYPE_CHECKING

from .clib.libket import Block
from .clib.wrapper import LibketError
from ._verification import _block_proprieties, _touched_qubits

if TYPE_CHECKING:
    from .base import Process, Quant


def _mark(process: Process, block: Block):
    """Apply the diagonal/permutation annotation of the enclosing builders."""
    if process._is_diagonal > 0:
        block.set_as_diagonal()
    elif process._is_permutation > 0:
        block.set_as_permutation()


def append_controlled(process: Process, body: Block, control_qubits: Quant):
    """Append ``body`` controlled by two or more ``control_qubits``.

    If at least two gates of ``body`` receive the controls, the controls are
    shared through a clean auxiliary qubit:

    .. code-block:: text

        compute:   aux ^= AND(control_qubits)   (relative-phase decomposition allowed)
        action:    body controlled by aux
        uncompute: adjoint of compute

    The action only reads ``aux`` (as a control) and does not touch the control
    qubits, so ``aux`` returns to |0> and the approximated decomposition of the
    compute step is exact (Theorems 1 to 3). Each gate of the body then needs a
    single control instead of the whole control set. Otherwise (a single gate,
    a body that acts on a control qubit, or no qubit left for the auxiliary),
    the controls are added to every gate of ``body``.

    The auxiliary qubit comes from a pool reserved for shared controls (see
    :meth:`~ket.base.Process._alloc_control_aux`), which the verification of
    :func:`~ket.operations.around` ignores: it is |0> outside the three steps
    above, which are always appended together.
    """
    aux = None
    if body.num_controllable_gates().value >= 2 and not (
        _touched_qubits(_block_proprieties(body)) & set(control_qubits.qubits)
    ):
        try:
            aux = process._alloc_control_aux()
        except LibketError:
            aux = None

    if aux is None:
        controlled = body.control(control_qubits.qubits)
        _mark(process, controlled)
        process.append_block(controlled)
        return

    try:
        target = Block(process)
        target.append_gate("PauliX", aux.qubits[0])
        compute = target.control(control_qubits.qubits)
        _mark(process, compute)
        compute.lock_control()
        compute.enable_approximated_decomposition()
        uncompute = compute.inverse()

        action = body.control(aux.qubits)
        _mark(process, action)

        process.append_block(compute, check_qubits=False)
        process.append_block(action)
        process.append_block(uncompute, check_qubits=False)
    finally:
        process._free_control_aux(aux)
