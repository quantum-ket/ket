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
from typing import Iterable, Optional
from functools import reduce
from random import choices

__all__ = ['quant', 'future', 'dump', 'dump_measure', 'metrics', 'context']

class quant(_quant):
    r"""Qubit list

    Allocate ``size`` qubits in the state :math:`\left|0\right>` and return
    its reference in a new :class:`~ket.types.quant`.

    Qubits allocated using a with statement must be released at the end of the
    scope.
    
    **Example:**
    
    .. code-block:: ket

        a = H(quant()) 
        b = X(quant())
        with quant() as aux: 
            with around(H, aux):
                with control(aux):
                    swap(a, b)
            result = measure(aux)
            if result == 1:
                X(aux) 
            aux.free() 
    
    :param size: The number of qubits to allocate.
    """

    def __init__(self, size : int = 1):
        r"""Allocate `size` qubits"""
        super().__init__(size)

    def __or__(self, other):
        ret = super().__or__(other)
        ret.__class__ = quant
        return ret

    def at(self, index : Iterable[int]):
        r"""Return qubits at `index`
        
        Create a new :class:`~ket.types.quant` with the qubit references at the
        position defined by the ``index`` list.

        **Example:**

        .. code-block:: ket

            q = quant()        
            odd = q.at(range(1, len(q), 2)) # = q[1::2]

        :param index: List of indexes.
        """

        return reduce(lambda a, b : a | b, [self[i] for i in index])   

    def __getitem__(self, param):
        ret = super().__getitem__(param)
        ret.__class__ = quant
        return ret

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

dump.__doc__ = """Dump qubits

Create a dump with the state current state of the :class:`~ket.types.quant`
``q``.

Creating a dump does not trigger quantum execution, but gathering any
information from it does.

**Example:**

.. code-block:: ket

    a = H(quant())
    b = ctrl(a, X, quant())
    d1 = dump(a|b)
    Y(a)
    d2 = dump(a|b)

    print(d1.show())
    print(d2.show())

:param q: Qubits to dump.
"""

class dump_measure():

    def __init__(self, q : quant):
        self.dump = dump(q)
        self.states = None
        self.probabilities = None
        self.sorted = False

    def get(self, n : Optional[int] = None, no_repeat : bool = False):

        if self.states is None:
            self.states = self.dump.get_states()
            self.probabilities = [self.dump.probability(i) for i in self.states]
        
        if n is None:
            return choices(self.states, weights=self.probabilities)[0]
        elif no_repeat:
            if not self.sorted:
                self.sorted = True
                sorted_states = [(s, p) for s, p in sorted(zip(self.states, self.probabilities), key = lambda x : x[1], reverse=True)]
                self.states = [s for s, _ in sorted_states]    
                self.probabilities = [p for _, p in sorted_states]
            
            return self.states[:n if n <= len(self.states) else None]
        else:
            return choices(self.states, weights=self.probabilities, k=n)
            