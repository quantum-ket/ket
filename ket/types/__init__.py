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

from ..ket import quant, future, dump, metrics
from ..gates import cnot, X
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
        X(self)

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

