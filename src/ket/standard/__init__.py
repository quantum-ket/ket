from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from functools import reduce
from operator import add
from ..base import base_measure, quant, future, qc_int, base_X
from ..preprocessor.statements import _ket_if, _ket_next  # pylint: disable=no-name-in-module
from .ctrl import ctrl, control
from .adj import around, adj, inverse

__all__ = ['inverse', 'control', 'ctrl',
           'adj', 'around', 'measure', 'qc_int']


def measure(qubits: quant | list[quant], free: bool = False) -> future | list[future]:
    """Quantum measurement

    Measure the qubits of a :class:`~ket.libket.quant` and return a
    :class:`~ket.libket.future` or [:class:`~ket.libket.future`].


    When measuring more than 64 qubits, Ket split the measure every
    63 qubits.

    Args:
        q: Qubits to measure.
        free: If ``True``, free the qubits after the measurement.
    """
    qubits = reduce(add, qubits) if len(qubits) else qubits

    ret = base_measure(qubits)
    if free:
        for i in qubits:
            end = _ket_if(base_measure(i))
            base_X(i)
            _ket_next(end)
        qubits.free()
    return ret
