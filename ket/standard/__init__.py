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

from ..ket import (quant, measure, report, exec_quantum, ctrl_begin, ctrl_end,
        adj_begin, adj_end, process_begin, process_end)
from typing import Iterable, Callable

__all__ = ['run', 'inverse', 'control', 'ctrl', 'adj', 'report', 'measure', 'exec_quantum']

class run:
    """Execute in a new process
    
    Run the quantum operations in a new separated process.
    
    **Usage:**

    .. code-block:: ket

        with run():
            ...

    """
    def __enter__ (self):
        process_begin()
    
    def __exit__ (self, type, value, tb):
        process_end()

class inverse:
    """Execute inverse

    Apply the quantum operations backwards.
    
    **Usage:**

    .. code-block:: ket

        with inverse():
            ...
            
    """
    def __enter__ (self):
        adj_begin()     

    def __exit__ (self, type, value, tb):
        adj_end()     

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

def ctrl(control : quant, func : Callable , *args, **kwargs):
    """Add qubits of controll to a operation call."""
    ctrl_begin(control)
    ret = func(*args, **kwargs)
    ctrl_end()
    return ret

def adj(func : Callable, *args, **kwargs):
    """Call the inverse of a quantum operation."""
    adj_begin()
    ret = func(*args, **kwargs)
    adj_end()
    return ret

