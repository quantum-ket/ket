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

from ..ket import quant as _quant, future, dump, metrics, context, measure
from ..gates import cnot, X
from ..standard import adj
from typing import Iterable

__all__ = ['quant', 'future', 'dump', 'metrics', 'context']

class quant(_quant):
    class quant_iter:
        def __init__(self, q):
            self.q = q
            self.idx = -1
            self.size = len(q)

        def __next__(self): 
            self.idx += 1
            if self.idx < self.size:
                return self.q[self.idx]
            raise StopIteration


    def at(self, index : Iterable[int]):
        """Return a quant with the qubits from index."""

        index = list(index)
        if len(index) == 0:
            return None
        else:
            q = self[index[0]]
            for i in index[1:]:
                q |= self[i]
            return q 
    
    def __iter__(self):
        return self.quant_iter(self)

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, tb):
        if not self.is_free():
            raise RuntimeError('non-free quant at the end of scope')

    def __int__(self):
        return measure(self).get()

    def __repr__(self):
        return '<Ket quant; '+str(len(self))+' qubits; '+self.this.__repr__()+'>'


def __future__setattr__(self, name, value):
    if name == 'set':
        self.set(value)
    else:
        self.__dict__[name] = value

future.__setattr__ = __future__setattr__

future.__int__ = lambda self : self.get()

future.__repr__ = lambda self : '<Ket future; '+self.this.__repr__()+'>'
dump.__repr__ = lambda self : '<Ket dump; '+self.this.__repr__()+'>'
metrics.__repr__ = lambda self : '<Ket metrics; '+self.this.__repr__()+'>'

