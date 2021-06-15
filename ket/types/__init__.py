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

from ..ket import quant as _quant, future, dump, measure
from typing import Iterable, Optional
from functools import reduce
from random import choices

__all__ = ['quant', 'future', 'dump', 'dump_measure']

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
        ret = super().dirty(size)
        ret.__class__ = quant
        return ret

    def __add__(self, other):
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

        return reduce(lambda a, b : a | b, [self[i] for i in index])   
    
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
        return '<Ket quant; '+str(len(self))+' qubits; '+super().__repr__()+'>'


future.__doc__ = \
    """Quantum computer's classical value

    Store a reference to a 64-bits integer available in the quantum computer. 

    The integer value are available to the classical computer only after the
    quantum execution.
    
    The follwing binary operations are available between
    :class:`~ket.types.future` variables and ``int``: 
        
        ``==``, ``!=``, ``<``, ``<=``,
        ``>``, ``>=``, ``+``, ``-``, ``*``, ``/``, ``<<``, ``>>``, ``and``,
        ``xor``, and ``or``.

    A :class:`~ket.types.future` variable is created with a quantum
    :func:`~ket.standard.measure` (1) , binary operation with a
    :class:`~ket.types.future` (2), or directly initialization with a ``int``
    (2).

    .. code-block:: ket

        q = H(quant(2))
        a = measure(q) # 1
        b = a*3        # 2
        c = future(42) # 3
        
    To pass an assignment expression to the quantum computer use the syntax:
    
    .. code-block:: ket

        <var>.set = <exp>

    :Example:

    .. code-block:: ket

        q = quant(2)
        with quant() as aux:
            done = future(False)
            while done != True:
                H(q)
                ctrl(q, X, aux)
                res = measure(aux)
                if res == 0:
                    done.set = True
                else:
                    X(q+aux)
            aux.free() 

    :type value: ``int``
    :param value: Classical integer to pass to the quantum computer.
    """

future.get.__doc__ = \
    """ Get the value from the quantum computer

    Note: 
        This method triggers the quantum execution.

    :Example:
    
    .. code-block:: ket

        q = H(quant(60))
        m = measure(q)
        print('result:', m.get())

    :rtype: ``int``
    """

future.set.__doc__ = \
    """Set the value

    :type other: :class:`~ket.types.future`
    :param other: New value.
    """ 
    
future.__radd__     = lambda self, other : (future(other) if not isinstance(other, future) else other) + self
future.__rsub__     = lambda self, other : (future(other) if not isinstance(other, future) else other) - self
future.__rmul__     = lambda self, other : (future(other) if not isinstance(other, future) else other) * self
future.__rtruediv__ = lambda self, other : (future(other) if not isinstance(other, future) else other) // self
future.__rdiv__     = future.__rtruediv__
future.__rlshift__  = lambda self, other : (future(other) if not isinstance(other, future) else other) << self
future.__rrshift__  = lambda self, other : (future(other) if not isinstance(other, future) else other) >> self
future.__rand__     = lambda self, other : (future(other) if not isinstance(other, future) else other) & self
future.__rxor__     = lambda self, other : (future(other) if not isinstance(other, future) else other) ^ self
future.__ror__      = lambda self, other : (future(other) if not isinstance(other, future) else other) | self

def _future_setattr(self, name, value):
    if name == 'set':
        self.set(value if isinstance(value, future) else future(value))
    else:
        self.__dict__[name] = value

future.__setattr__ = _future_setattr
future.__int__ = lambda self : self.get()
future.__repr__ = lambda self : '<Ket future; '+self.this.__repr__()+'>'


dump.__repr__ = lambda self : '<Ket dump; '+self.this.__repr__()+'>'

dump.__doc__ = \
    """Dump qubits

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

def _dump_show(self, format : Optional[str] = None) -> str:
    """Return the quantum state as a string

    :type format: ``str``
    :param format: Format string that matchs ``(i|b)\d*(:(i|b)\d*)*``.
    """
    
    return self._show(format if format is not None else "")

dump.show = _dump_show

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
            