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

from ..libket import measure as _measure, process_begin, process_end, quant, future, exec_quantum, qc_int
from ..preprocessor import _ket_if, _ket_next
from .ctrl import *
from .adj import *
from functools import reduce
from operator import add

__all__ = ['run', 'inverse', 'control', 'ctrl', 'adj', 'around', 'measure', 'exec_quantum', 'qc_int']

def measure(q : quant | [quant], free : bool = False) -> future:
    """Quantum measurement

    Measure the qubits of a :class:`~ket.libket.quant` and return a
    :class:`~ket.libket.future`.

    Args:
        q: Qubits to measure.
        free: If ``True``, free the qubits after the measuremet.
    """
    q = reduce(add, q)

    ret = _measure(q)
    if free:
        for i in q:
            end = _ket_if(_measure(i))
            X(i)
            _ket_next(end) 
        q.free()
    return ret

class run:
    """Execute in a new process
    
    Run the quantum operations in a new separated process.
    
    :Usage:

    .. code-block:: ket

        with run():
            ...

    """
    def __enter__ (self):
        process_begin()
    
    def __exit__ (self, type, value, tb):
        process_end()
