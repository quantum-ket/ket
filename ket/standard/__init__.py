from __future__ import annotations
#  Copyright 2020, 2021 Evandro Chagas Ribeiro da Rosa <evandro.crr@posgrad.ufsc.br>
#  Copyright 2020, 2021 Rafael de Santiago <r.santiago@ufsc.br>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from ..base import base_measure, quant, future, qc_int, base_X
from ..preprocessor import _ket_if, _ket_next
from .ctrl import *
from .adj import *
from functools import reduce
from operator import add

__all__ = ['inverse', 'control', 'ctrl',
           'adj', 'around', 'measure', 'qc_int']


def measure(q: quant | list[quant], free: bool = False) -> future | list[future]:
    """Quantum measurement

    Measure the qubits of a :class:`~ket.libket.quant` and return a
    :class:`~ket.libket.future` or [:class:`~ket.libket.future`].


    When measuring more than 64 qubits, Ket split the measure every
    63 qubits.

    Args:
        q: Qubits to measure.
        free: If ``True``, free the qubits after the measurement.
    """
    q = reduce(add, q) if len(q) else q

    ret = base_measure(q)
    if free:
        for i in q:
            end = _ket_if(base_measure(i))
            base_X(i)
            _ket_next(end)
        q.free()
    return ret
