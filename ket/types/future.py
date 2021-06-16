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

from ..ket import future, measure

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
future.__repr__ = lambda self : f"<Ket type 'future'; {self.this.__repr__()}>"
