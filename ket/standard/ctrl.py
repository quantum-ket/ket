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

from ..ket import quant, ctrl_begin, ctrl_end
from typing import Iterable, Union, Callable

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

    def __init__(self, *ctr : Iterable[quant]):
        self.ctr = ctr

    def __enter__ (self):
        for c in self.ctr:
            ctrl_begin(c)
     
    def __exit__ (self, type, value, tb):
        for _ in self.ctr:
            ctrl_end()

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
