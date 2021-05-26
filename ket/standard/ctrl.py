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
from typing import Iterable, List, Union, Callable
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
    r"""Add controll-qubits to a Callable

    **Call with control qubits**

    * ``control`` must be :class:`~ket.types.quant` or :class:`~ket.types.quant` iterable.
    * ``func`` must be ``Callable`` or ``Callable`` iterable.

    .. code-block:: ket

        ret1 = ctrl(control_qubits, func, *args, **kwargs)
        # Equivalent to:
        # with control(control_qubits):
        #     ret1 = func(*args, **kwargs)

        ret2 = ctrl([q0, q1, q2, ...], func, *args, **kwargs)
        # Equivalent to:
        # with control(*control):
        #     ret2 = func(*args, **kwargs)
        
        ret3 = ctrl(control_qubits, [f0, f1, f2, ...], *args, **kwargs)
        # Equivalent to:
        # ret3 = []
        # with control(control_qubits):
        #     for f in func:
        #     ret3.append(f(*args, **kwargs))
 
    **Create controlled-operation**

    1. If the keyword argument ``later_call`` is ``True``, return a
    ``Callable[[], Any]``:

    .. code-block:: ket

        ctrl_func = ctrl(control_qubits, func, *args, **kwargs, later_call=True)
        # Equivalent to:
        # ctrl_func = lambda : ctrl(control_qubits, func, *args, **kwargs)

    **Example:**

    .. code-block:: ket
        :emphasize-lines: 8

        def increment(q):
            if len(q) > 1:
                ctrl(q[-1], increment, q[:-1])
            X(q[-1])

        size = ceil(log2(len(inputs)))+1
        with quant(size) as inc:
            with around(ctrl(q , increment, inc, later_call=True) for q in inputs):
                with control(inc, on_state=len(inputs)//2):
                    X(output)
            inc.free()

    2. If ``control`` and ``target`` qubits are ``int``, ``slice``, or
    ``Iterable[int]``,  return a ``Callable[[quant], Any]``:

    .. code-block:: ket

        ctrl_func = ctrl(ctrl_index, func, target_index)
        # Equivalent to:
        # ctrl_func = lambda q : ctrl(q[ctrl_index], func, q[target_index])

    **Example:**

    .. code-block:: ket

        with around(ctrl(0, X, slice(1, None)), q): # ctrl(q[0], X, q[1:]) 
            H(q[0])                                 # H(q[0])
                                                    # ctrl(q[0], X, q[1:])
 
    Args:
        control: Control qubits, use :class:`~ket.types.quant` for call with control qubits and index to create controlled-operations.
        func: Function or list of functions to add control.

    Other Parameters:
        target (``Union[slice, int, Iterable[int]]``): Target qubits to create controlled-operations.

    Keyword Args:
        later_call (``bool``): If ``True``, do not execute and return a ``Callable[[], Any]``.
    """

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
