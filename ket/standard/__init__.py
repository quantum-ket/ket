# MIT License
# 
# Copyright (c) 2020 Evandro Chagas Ribeiro da Rosa <evandro.crr@posgrad.ufsc.br>
# Copyright (c) 2020 Rafael de Santiago <r.santiago@ufsc.br>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
    func(*args, **kwargs)
    adj_end()

