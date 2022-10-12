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

from functools import reduce
from operator import add
from ..base import quant


class QuantumGate:  # pylint: disable=missing-class-docstring
    _gates_1 = []
    _gates_m = []
    _matrix = {}
    _effect = {}

    @classmethod
    @property
    def __doc__(cls):
        new_line = '\n'
        return \
            f"""

Single Qubit Gates
~~~~~~~~~~~~~~~~~~

.. csv-table::
    :delim: ;
    :header: Gate,  Function, Matrix, Effect

{new_line.join(f'    {gate}; {func}; {cls._matrix[gate]}; {cls._effect[gate]}' for gate, func in cls._gates_1)}

Multiple Qubit Gates
~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :delim: ;
    :header: Gate,  Function, Matrix

{new_line.join(f'    {gate}; {func}; {cls._matrix[gate]}' for gate, func in cls._gates_m)}

"""  # pylint: disable=line-too-long

    def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
        if 'doc' in kwargs:
            doc = kwargs['doc']
            name = kwargs['name']
            if 'q_args' in kwargs and kwargs['q_args'] == 1 or 'q_args' not in kwargs:
                cls._gates_1.append((name, doc['func']))
                cls._effect[name] = doc['effect']
            else:
                cls._gates_m.append((name, doc['func']))
            cls._matrix[name] = doc['matrix']

        return super().__new__(cls)

    def __init__(self, *, name, gate, c_args=0, q_args=1, **kwargs):  # pylint: disable=unused-argument
        self.name = name
        self.gate = gate
        self.c_args = c_args
        self.q_args = q_args

    def __call__(self, *args):
        args_size = len(args)
        if args_size < self.c_args:
            raise ValueError(
                f'{self.name} requirers {self.c_args} classical parameters, {args_size} provided')
        if args_size == self.c_args:
            if any(isinstance(arg, quant) for arg in args):
                raise ValueError(
                    f'{self.name} requirers {self.c_args} classical parameters')

            args_name = ', '.join(str(arg) for arg in args)
            return QuantumGate(
                name=f'{self.name}({args_name})',
                gate=lambda *q_args: self.__call__(*args, *q_args),
                q_args=self.q_args
            )

        c_args = args[:self.c_args]
        q_args = args[self.c_args:]

        if all(isinstance(arg, QuantumGate) for arg in q_args):
            q_args_size = 0
            for arg in q_args:
                q_args_size += arg.q_args
            if q_args_size != self.q_args:
                raise ValueError(
                    f'{self.name} requirers {self.q_args} quantum parameters, {q_args_size} provided')  # pylint: disable=line-too-long

            def new_gate(*args):
                args_ = args
                for gate in q_args:
                    gate(*args_[:gate.q_args])
                    args_ = args_[gate.q_args:]
                return self.gate(*c_args, *args)

            args_name = ', '.join(gate.name for gate in q_args)
            return QuantumGate(
                name=f'{self.name}({args_name})',
                gate=new_gate,
                q_args=self.q_args)
        if any(isinstance(arg, QuantumGate) for arg in q_args):
            raise ValueError('Incomplete gate call')

        reduced_q_args = tuple(
            map(lambda q: reduce(add, q) if len(q) else q, q_args))
        self.gate(*c_args, *reduced_q_args)
        return reduced_q_args[0] if len(reduced_q_args) == 1 else reduced_q_args

    def __repr__(self):
        return f"<Ket '{self.name} Gate'>"
