from __future__ import annotations
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

from ..ket import dump as _dump
from .quant import quant
from typing import Optional
from random import choices

class dump(_dump):
    """Dump qubits

    Create a dump with the state current state of the :class:`~ket.types.quant`
    ``q``.

    Creating a dump does not trigger quantum execution, but gathering any
    information from it does.

    :Example:

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
    
    def __init__(self, q : quant):
        super().__init__(q)
        self.size = len(q)
    
    def get_states(self) -> list[int]:
        """Return the list of basis states."""
        return super.get_states()

    def amplitude(self, state : int) -> complex | list[complex]:
        """Return the amplitude of a given state
        
        If the state is entangled with a not dumped qubit, return a list of
        amplitudes.
        
        Args:
            state: Basis state.
        """

        ret = super().amplitude(state)
        return ret[0] if len(ret) == 1 else ret
    
    def probability(self, state : int) -> float:
        """Return the measure probabilit of a basis state.
          
        Args:
            state: Basis state.
        """
        return super().probability(state)

    def show(self, format : Optional[str] = None) -> str:
        r"""Return the quantum state as a string
        
        Use the format starting to change the print format of the basis states:

        * ``i``: print the state in the decimal base
        * ``b``: print the state in the binary base (default)
        * ``i|b<n>``: separate the ``n`` first qubits, the remaining print in the binary base
        * ``i|b<n1>:i|b<n2>[:i|b<n3>...]``: separate the ``n1, n2, n3, ...`` first qubits
        
        :Example:
            
        .. code-block:: ket

            q = quant(20)
            X(ctrl(H(q[0]), X, q[1:])[1::2])
            d = dump(q)

            print(d.show('i'))
            #|174762⟩		(50%)
            #0.707107                 ≅  1/√2
            #
            #|873813⟩		(50%)
            #0.707107                 ≅  1/√2
            print(d.show('b'))
            #|00101010101010101010⟩		(50%)
            #0.707107                 ≅  1/√2
            #
            #|11010101010101010101⟩		(50%)
            #0.707107                 ≅  1/√2
            print(d.show('i4'))
            #|2⟩|1010101010101010⟩		(50%)
            #0.707107                 ≅  1/√2
            #
            #|13⟩|0101010101010101⟩		(50%)
            #0.707107                 ≅  1/√2
            print(d.show('b5:i4'))
            #|00101⟩|2⟩|101010101010⟩		(50%)
            #0.707107                 ≅  1/√2
            #
            #|11010⟩|5⟩|010101010101⟩		(50%)
            #0.707107                 ≅  1/√2

        Args:
            format: Format string that matchs ``(i|b)\d*(:(i|b)\d+)*``.
        """
        
        if format == 'i' or format == 'b':
            format += str(self.size)

        return super().show(format if format is not None else "")

    __repr__ = lambda self : f"<Ket type 'dump'; {super().__repr__()}>"

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
 