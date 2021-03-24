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

__all__ = ['run', 'inverse', 'control', 'ctrl', 'adj', 'report', 'measure', 'exec_quantum']

class run:
    """Run the quantum operations in a new process.
    
    Usage:

    .. code-block:: ket

        with run():
            ...

    """
    def __enter__ (self):
        process_begin()
    
    def __exit__ (self, type, value, tb):
        process_end()

class inverse:
    """Apply the quantum operations backwards.
    
    Usage:

    .. code-block:: ket

        with inverse():
            ...
            
    """
    def __enter__ (self):
        adj_begin()     

    def __exit__ (self, type, value, tb):
        adj_end()     

class control:
    """Apply controlled quantum operations.
    
    Usage:

    .. code-block:: ket
    
        with control(cont):
            ...
            
    """
    def __init__(self, c : quant):
        self.c = c

    def __enter__ (self):
        ctrl_begin(self.c)
     
    def __exit__ (self, type, value, tb):
        ctrl_end()

def ctrl(control : quant, func, *args, **kwargs):
    """Add qubits of controll to a operation call."""
    ctrl_begin(control)
    ret = func(*args, **kwargs)
    ctrl_end()
    return ret

def adj(func, *args, **kwargs):
    """Call the inverse of a quantum operation."""
    adj_begin()
    func(*args, **kwargs)
    adj_end()

