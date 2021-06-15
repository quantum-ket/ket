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

from ..ket import measure as _measure, exec_quantum, process_begin, process_end
from ..types import quant, future
from ..preprocessor import _ket_if, _ket_next
from .ctrl import *
from .adj import *

__all__ = ['run', 'inverse', 'control', 'ctrl', 'adj', 'around', 'measure', 'exec_quantum']

exec_quantum.__doc__ = \
    """Trigger the quantum execution

    Execute the top process and fullfil the :class:`~ket.types.future` and
    :class:`~ket.types.dump` variables.
    """

def measure(q : quant, free : bool = False) -> future:
    """Quantum measurement

    Measure the qubits of a :class:`~ket.types.quant` and return a
    :class:`~ket.types.future`.

    Args:
        q: Qubits to measure.
        free: If ``True``, free the qubits after the measuremet.
    """

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
