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

from ..ket import ctrl_begin, ctrl_end, X
from ..types import quant
from typing import Iterable, Union, Callable, Optional
from functools import reduce

class control:
    r"""Controlled execution
    
    Apply controlled quantum operations.

    **Usage:**

    .. code-block:: ket
    
        with control(ctr[, ...]):
            ...

    Note:
        Use ``~ctrl`` to control on :math:`\left|0\right>`.

    :param ctr: control :class:`~ket.type.quant` variables
    """

    def __init__(self, *ctr : Iterable[quant], on_state : Optional[int] = None):
        self.ctr = reduce(lambda a, b : a | b, ctr)
        self.on_state = on_state
        if on_state is not None:
            self.mask = [int(i) for i in ('{0:0'+str(len(self.ctr))+'b}').format(on_state)]

    def _apply_mask(self):
        for i, q in zip(self.mask, self.ctr):
            if i == 0:
                X(q)

    def __enter__ (self):
        if self.on_state is not None:
            self._apply_mask()
        ctrl_begin(self.ctr)
     
    def __exit__ (self, type, value, tb):
        ctrl_end()
        if self.on_state is not None:
            self._apply_mask()

def ctrl(control : Union[Iterable[quant], quant, slice, int], func : Union[Callable, Iterable[Callable]] , *args, **kwargs):
    """Add qubits of controll to a operation call."""

    if type(control) == slice or type(control) == int:
        if 'target' not in kwargs:
            raise ValueError("Keyword argument 'target' no provided")
        return lambda q : ctrl(q[control], func, q[kwargs['target']])

    ret = []
    if hasattr(control, '__iter__'):
        for c in control:
            ctrl_begin(c)
    else:
        ctrl_begin(control)

    if hasattr(func, '__iter__'):
        for f in func:
            ret.append(f(*args, **kwargs))
    else:
        ret = func(*args, **kwargs)

    if hasattr(control, '__iter__'):
        for _ in control:
            ctrl_end()
    else:
        ctrl_end()
    
    return ret
