#  Copyright 2021 Evandro Chagas Ribeiro da Rosa <evandro.crr@posgrad.ufsc.br>
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

from functools import reduce
from operator import add

class quantum_gate:
    _gates = []
    _matrix = {}
    _effect = {}

    @classmethod
    @property
    def __doc__(cls):
        nl = '\n'
        return \
f"""
.. csv-table::
    :header: Gate,  Function, Matrix, Effect

{nl.join(f'    {gate}, {func}, {cls._matrix[gate]}, {cls._effect[gate]}' for gate, func in cls._gates)}

"""

    def __new__(cls, *args, **kwargs):
        if 'name' in kwargs:
            name = kwargs['name']
            cls._gates.append((name, kwargs['func']))
            cls._matrix[name] = kwargs['matrix']
            cls._effect[name] = kwargs['effect']
        return cls

    def __init__(self, *, name, gate, c_args=0, q_args=1, **kwargs):
        self.name = name
        self.gate = gate
        self.c_args = c_args
        self.q_args = q_args

    def __call__(self, *args):
        args_size = len(args)
        if args_size < self.c_args:
            raise ValueError(f'{self.name} requirers {self.c_args} classical parameters, {args_size} provided')
        elif args_size == self.c_args:
            if any(isinstance(arg, quant) for arg in args):
                raise ValueError(f'{self.name} requirers {self.c_args} classical parameters')

            args_name = ' '.join(str(arg) for arg in args)
            return quantum_gate(
                name=f'{self.name}({args_name})', 
                gate=lambda *q_args : self.__call__(*args, *q_args),
                q_args=self.q_args
            )

        c_args = args[:self.c_args]
        q_args = args[self.c_args:]

        if all(isinstance(arg, quantum_gate) for arg in q_args):
            q_args_size = 0
            for arg in q_args:
                q_args_size += arg.q_args
            if q_args_size != self.q_args:
                raise ValueError(f'{name} requirers {self.q_args} quantum parameters, {q_args_size} provided')

            def new_gate(*args):
                args_ = args
                for gate in q_args:
                    gate(*args_[:gate.q_args])
                    args_ = args_[gate.q_args:]
                return self.gate(*c_args, *args)

            args_name = ' '.join(gate.name for gate in q_args)
            return quantum_gate(
                name=f'{self.name}({args_name})', 
                gate=new_gate, 
                q_args=self.q_args)
        elif any(isinstance(args, quantum_gate) for arg in q_args):
            raise ValueError(f'Incomplete gate call')

        for i in range(len(q_args)):
            q_args[i] = reduce(add, q_args[i])

        self.gate(*c_args, *q_args)
        return q_args

    def __repr__(self):
        return f'{self.name} Ket Quantum Gate'
        
    