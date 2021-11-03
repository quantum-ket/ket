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
from functools import reduce
from operator import add

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
        q = reduce(add, q)
        super().__init__(q)
        self.size = len(q)
        padding = self.size%64
        if padding != 0:
            padding = 64-padding
        self._bit_size = self.size+padding
        self._states = None
    
    @property
    def states(self) -> list[int]:
        """List of basis states."""

        if self._states is None:
            self._states = [int(reduce(add, [f'{i:064b}' for i in reversed(state)]) , 2) for state in super().get_states()]

        return self._states

    def _convert_state(self, state : int) -> list[int]:
        state_bin = f'{state:0b}'
        padding = self._bit_size - len(state_bin)
        if padding != 0:
            state_bin = '0'*padding+state_bin    
        
        return [int(state_bin[i:i+64], 2) for i in reversed(range(0, self._bit_size, 64))]

    def amplitude(self, state : int) -> complex | list[complex]:
        """Return the amplitude of a given state
        
        If the state is entangled with a not dumped qubit, return a list of
        amplitudes.
        
        Args:
            state: Basis state.
        """

        ret = super().amplitude(self._convert_state(state))
        return ret[0] if len(ret) == 1 else ret
    
    def probability(self, state : int) -> float:
        """Return the measure probabilit of a basis state.
          
        Args:
            state: Basis state.
        """
        return super().probability(self._convert_state(state))

    def show(self, format : Optional[str] = None) -> str:
        r"""Return the quantum state as a string
        
        Use the format starting to change the print format of the basis states:

        * ``i``: print the state in the decimal base
        * ``b``: print the state in the binary base (default)
        * ``i|b<n>``: separate the ``n`` first qubits, the remaining print in the binary base
        * ``i|b<n1>:i|b<n2>[:i|b<n3>...]``: separate the ``n1, n2, n3, ...`` first qubits
        
        :Example:
            
        .. code-block:: ket

            q = quant(19)
            X(ctrl(H(q[0]), X, q[1:])[1::2])
            d = dump(q)

            print(d.show('i'))
            #|87381⟩         (50.00%)
            # 0.707107               ≅      1/√2
            #|436906⟩        (50.00%)
            # 0.707107               ≅      1/√2

            print(d.show('b'))
            #|0010101010101010101⟩   (50.00%)
            # 0.707107               ≅      1/√2
            #|1101010101010101010⟩   (50.00%)
            # 0.707107               ≅      1/√2
            
            print(d.show('i4'))
            #|2⟩|101010101010101⟩    (50.00%)
            # 0.707107               ≅      1/√2
            #|13⟩|010101010101010⟩   (50.00%)
            # 0.707107               ≅      1/√2

            print(d.show('b5:i4'))
            #|00101⟩|5⟩|0101010101⟩  (50.00%)
            # 0.707107               ≅      1/√2
            #|11010⟩|10⟩|1010101010⟩ (50.00%)
            # 0.707107               ≅      1/√2

        Args:
            format: Format string that matchs ``(i|b)\d*(:(i|b)\d+)*``.
        """
        
        dump_str = ''

        if format is not None:
            if format == 'b' or format == 'i':
                format += str(self.size)
            fmt = []
            count = 0
            for b, size in map(lambda f : (f[0], int(f[1:])), format.split(':')):
                fmt.append((b, count, count+size))
                count += size
            if count < self.size:
                fmt.append(('b', count, self.size))
        else:
            fmt = [('b', 0, self.size)]

        fmt_ket = lambda state, begin, end, f : f'|{state[begin:end]}⟩' if f == 'b' else f'|{int(state[begin:end], base=2)}⟩'

        for state in self.states:
            dump_str += ''.join(fmt_ket(f'{state:0{self.size}b}', b, e, f) for f, b, e in fmt)
            dump_str += f"\t({100*self.probability(state):.2f}%)\n"
            for amp in super().amplitude(self._convert_state(state)):
                real = abs(amp.real) > 1e-10
                real_l0 = amp.real < 0

                imag = abs(amp.imag) > 1e-10
                imag_l0 = amp.imag < 0

                sqrt_dem = 1/abs(amp)**2
                use_sqrt = abs(round(sqrt_dem)-sqrt_dem) < .001
                sqrt_dem = f'/√{round(1/abs(amp)**2)}'
 
                if real and imag:
                    sqrt_num = ('(-1' if real_l0 else ' (1')+('-i' if imag_l0 else '+i')
                    sqrt_str = f'\t≅ {sqrt_num}){sqrt_dem}\n' if use_sqrt else '\n'
                    dump_str += f"{amp.real:9.6f}{amp.imag:+.6f}i"+sqrt_str 
                elif real:
                    sqrt_num = '  -1' if real_l0 else '   1'
                    sqrt_str = f'\t≅   {sqrt_num}{sqrt_dem}\n' if use_sqrt else '\n'
                    dump_str += f"{amp.real:9.6f}       "+sqrt_str
                else:
                    sqrt_num = '  -i' if imag_l0 else '   i'
                    sqrt_str = f'\t≅   {sqrt_num}{sqrt_dem}\n' if use_sqrt else '\n'
                    dump_str += f" {amp.imag:17.6f}i"+sqrt_str

        return dump_str

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
 