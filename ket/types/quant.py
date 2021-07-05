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

from ..ket import quant as _quant, measure, quant_dirty
from typing import Iterable
from functools import reduce
from operator import add

class quant(_quant):
    r"""Qubit list

    Allocate ``size`` qubits in the state :math:`\left|0\right>` and return
    its reference in a new :class:`~ket.types.quant`.

    Qubits allocated using a ``with`` statement must be released at the end of
    the scope.
    
    Example:
    
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

    :Qubit Indexing:

    Use brackets to index qubits as in a ``list`` and use ``+`` to concatenate
    two :class:`~ket.types.quant`.

    Example:

    .. code-block:: ket

        q = quant(20)        
        head, tail = q[0], q[1:]
        init, last = q[:-1], q[-1]
        even = q[::2]
        odd = q[1::2]
        reverse = reversed(q) # invert qubits order

        
        a, b = quant(2) # |a⟩, |b⟩
        c = a+b         # |c⟩ = |a⟩|b⟩ 

    Args:
        size: The number of qubits to allocate.
    """

    def __init__(self, size : int = 1):
        r"""Allocate `size` qubits"""
        super().__init__(size)
        
    @staticmethod
    def dirty(size : int = 1):
        """Allocate dirty qubits

        Allocate ``size`` qubits at an unknown state. 

        warning:
            Use dirty qubits may have side effects due to previous entanglement.

        Args:
            size: The number of dirty qubits to allocate.
        """
        ret = quant_dirty(size)
        ret.__class__ = quant
        return ret

    def __add__(self, other):
        ret = super().__or__(other)
        ret.__class__ = quant
        return ret

    def __or__(self, other):
        ret = super().__or__(other)
        ret.__class__ = quant
        return ret

    def at(self, index : Iterable[int]):
        r"""Return qubits at `index`
        
        Create a new :class:`~ket.types.quant` with the qubit references at the
        position defined by the ``index`` list.

        :Example:

        .. code-block:: ket

            q = quant(20)        
            odd = q.at(range(1, len(q), 2)) # = q[1::2]

        Args:
            index: List of indexes.
        """

        return reduce(add, [self[i] for i in index])   
    
    def free(self, dirty : bool = False):
        r"""Free the qubits

        All qubits must be at the state :math:`\left|0\right>` before the call,
        otherwise set the ``dirty`` param to ``True``.

        Warning: 
            No check is applied to see if the qubits are at state
            :math:`\left|0\right>`.

        Args:
            dirty: Set ``True`` to free dirty qubits.
        """
 
        super().free(dirty)
        
    def is_free(self) -> bool:
        """Return ``True`` when all qubits are free"""
        return super().is_free()

    def __reversed__(self):
        ret = super().inverted()
        ret.__class__ = quant
        return ret

    def __getitem__(self, key):
        ret = super().__getitem__(key)
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
            return self
 
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
        return f"<Ket type 'quant'; {str(len(self))} qubits; {super().__repr__()}>"
