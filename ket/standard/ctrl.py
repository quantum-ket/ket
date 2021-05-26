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
from typing import Iterable, List, Union, Callable, Optional
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

    def __init__(self, *ctr : Iterable[quant], on_state : Union[int, List[int], None] = None):
        self.ctr = reduce(lambda a, b : a | b, ctr)
        self.on_state = on_state
        if on_state is not None:
            if hasattr(on_state, '__iter__'):
                self.mask = on_state
            else:
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
            
def _ctrl(control : Union[Iterable[quant], quant ], func : Union[Callable, Iterable[Callable]] , *args, **kwargs):
    ret = []
    if hasattr(control, '__iter__'):
        control = reduce(lambda a, b : a | b, control)

    ctrl_begin(control)

    if hasattr(func, '__iter__'):
        for f in func:
            ret.append(f(*args, **kwargs))
    else:
        ret = func(*args, **kwargs)

    ctrl_end()
    
    return ret

def _get_qubits_at(control):
    if type(control) == slice or type(control) == int:
        return lambda q : q[control]
    elif hasattr(control, '__iter__') and all(type(i) == int for i in control):
        return lambda q : q.at(control)
    else:
        return None

def ctrl(control : Union[Iterable[quant], quant, slice, int, Iterable[int]], func : Union[Callable, Iterable[Callable]] , *args, **kwargs):
    """Add qubits of controll to a operation call."""

    q_ctrl = _get_qubits_at(control)
    if q_ctrl is not None:
        if 'target' not in kwargs and not len(args):
            raise ValueError("Argument 'target' no provided")
        
        if 'target' in kwargs:
            if len(kwargs) > 1:
                raise TypeError("Invalid keyword arguments for 'ctrl'")
            elif len(args) == 1:
                raise ValueError("Argument 'target' provided twice")
            elif len(args):
                raise TypeError("Invalid arguments for 'ctrl'")
            target = kwargs['target'] 
        else:
            if len(args) > 1:
                raise TypeError("Invalid arguments for 'ctrl'")
            elif 'target' in kwargs:
                raise ValueError("Argument 'target' provided twice")
            elif len(kwargs):
                raise TypeError("Invalid keyword arguments for 'ctrl'")
            target = args[0]
            
        q_trgt = _get_qubits_at(target)

        def _ctrl_gate(q : quant) -> quant:
            _ctrl(q_ctrl(q), func, q_trgt(q))
            return q
        return _ctrl_gate

    if 'later_call' in kwargs and kwargs['later_call']:
        del kwargs['later_call']
        return lambda : _ctrl(control, func, *args, **kwargs)
    else:
        return _ctrl(control, func, *args, **kwargs)
