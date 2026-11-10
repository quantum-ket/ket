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
from .clib.log import get_logger
from .clib.wrapper import LibketError
from ._verification import _block_proprieties, _touched_qubits

if TYPE_CHECKING:
    from .base import Process, Quant

_LOGGER = get_logger(__name__)


def _mark(process: Process, block: Block):
    """Apply the diagonal/permutation annotation of the enclosing builders."""
    if process._is_diagonal > 0:
        block.set_as_diagonal()
    elif process._is_permutation > 0:
        block.set_as_permutation()


def append_controlled(process: Process, body: Block, control_qubits: Quant):
    r"""Append ``body`` controlled by two or more ``control_qubits``.

    If sharing reduces the CNOT count (estimated by Libket, see below), the
    controls are shared through a clean auxiliary qubit:

    .. code-block:: text

        compute:   aux ^= AND(control_qubits)   (relative-phase decomposition allowed)
        action:    body controlled by aux
        uncompute: adjoint of compute

    The action only reads ``aux`` (as a control) and does not touch the control
    qubits, so ``aux`` returns to |0> and the approximated decomposition of the
    compute step is exact. Each gate of the body then needs a single control
    instead of the whole control set. Otherwise (no CNOT saved,
    no decomposition of the gates, a body that acts on a control qubit, or no
    qubit left for the auxiliary), the controls are added to every gate of
    ``body``.

    The estimate is computed by ``Process::shared_control_gain`` in Libket.
    Sharing costs the compute and the uncompute of the approximated AND of the
    controls. It saves, for each gate of the body,
    the difference between its cost with all the controls and with one, and a
    non-zero global phase saves its correction, a :math:`C^{n-1}P` gate on the
    controls, which becomes a :math:`P` gate on ``aux``. Libket obtains each
    cost from the decomposition it would select with the qubits free now, so
    the decision takes into account the auxiliary qubits available to the
    decompositions, including the one that the shared control takes. The
    controls are never shared when the process does not decompose the gates,
    since the shared form would only add a qubit and gates to the circuit.

    The auxiliary qubit comes from a pool reserved for shared controls (see
    :meth:`~ket.base.Process._alloc_control_aux`), which the verification of
    :func:`~ket.operations.around` ignores: it is |0> outside the three steps
    above, which are always appended together.
    """
    aux = None
    num_controls = len(control_qubits)
    touches_controls = bool(
        _touched_qubits(_block_proprieties(body)) & set(control_qubits.qubits)
    )
    gain = 0
    if touches_controls:
        _LOGGER.debug(
            "not sharing the controls %s: the block acts on a control qubit",
            control_qubits.qubits,
        )
    else:
        gain = process.shared_control_gain(
            body, num_controls, not process._control_aux_free
        ).value
        if gain > 0:
            try:
                aux = process._alloc_control_aux()
            except LibketError:
                _LOGGER.debug(
                    "not sharing the controls %s: no qubit left for the auxiliary",
                    control_qubits.qubits,
                )
        else:
            _LOGGER.debug(
                "not sharing the controls %s: it would not save CNOTs (gain %d)",
                control_qubits.qubits,
                gain,
            )

    if aux is None:
        controlled = body.control(control_qubits.qubits)
        _mark(process, controlled)
        process.append_block(controlled)
        return

    _LOGGER.info(
        "sharing %d controls %s through the auxiliary qubit %d (saves about %d CNOTs)",
        num_controls,
        control_qubits.qubits,
        aux.qubits[0],
        gain,
    )

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
