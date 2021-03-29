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

from ..ket import quant, future, dump, metrics
from ..gates import cnot, x
from ..standard import adj
from typing import Iterable

__all__ = ['quant', 'future', 'dump', 'metrics']

def __quant__at__(self, index : Iterable[int]):
    """Return a quant with the qubits from index."""

    index = list(index)
    if len(index) == 0:
        return None
    else:
        q = self[index[0]]
        for i in index[1:]:
            q |= self[i]
        return q 

quant.at = __quant__at__

class quant_invert(quant):
    def __init__(self, q : quant):
        super().__init__(len(q))
        self.base_quant = q
        self.prepare()
    
    def __del__(self):
        adj(self.prepare)
        self.free()
        
    def __or__(self, other):
        raise RuntimeError("cannot concatenate 'quant_invert'")

    def prepare(self):
        cnot(self.base_quant, self)
        x(self)

quant.__invert__ = lambda self : quant_invert(self)

class __quant__iter__:
    def __init__(self, q):
        self.q = q
        self.idx = -1
        self.size = len(q)

    def __next__(self): 
        self.idx += 1
        if self.idx < self.size:
            return self.q[self.idx]
        raise StopIteration

quant.__iter__ = lambda self : __quant__iter__(self)

quant.__enter__ = lambda self : self

def __quant__exit__(self, type, value, tb):
    if not self.is_free():
        raise RuntimeError('non-free quant at the end of scope')

quant.__exit__ = __quant__exit__

quant.__repr__ = lambda self : '<Ket quant; '+str(len(self))+' qubits; '+self.this.__repr__()+'>'
future.__repr__ = lambda self : '<Ket future; '+self.this.__repr__()+'>'
dump.__repr__ = lambda self : '<Ket dump; '+self.this.__repr__()+'>'
metrics.__repr__ = lambda self : '<Ket metrics; '+self.this.__repr__()+'>'

