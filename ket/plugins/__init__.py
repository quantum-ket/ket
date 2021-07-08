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

from ..ket import plugin
from ..types import quant
from inspect import getsource
from base64 import b64encode
from textwrap import dedent
from typing import Optional

__all__ = ['plugin', 'pown', 'diagonal', 'make_quantum']

def pown(a : int, x : quant, N : int) -> quant: 
    r"""Apply a modular exponentiation in a superposition.

    .. math::

        \left| x \right> \left| 0 \right> \rightarrow  \left| x \right> \left| a^x\; \text{mod} \, N \right>

    Args:
        a: :math:`a`.
        x: :math:`x`.
        N: :math:`N`.

    :return: :class:`~ket.types.quant` with the operation result.
    """
    
    ret = quant(N.bit_length())
    arg = str(len(ret)) + ' ' + str(a) + ' ' + str(N)

    plugin('ket_pown', x|ret, arg)

    return ret

def diagonal(diag : list[float], q : quant):
    r"""Apply a diagonal matrix.

    .. math::

        \begin{bmatrix}
                 e^{i\lambda_0} & 0              & \cdots & 0 \\
                 0              & e^{i\lambda_1} & \cdots & 0 \\
                 \vdots         & \vdots         & \ddots & \vdots \\
                 0              & 0              & \cdots & e^{i\lambda_{2^n-1}} 
             \end{bmatrix}

    Args:
        diag: :math:`\left[ \lambda_0, \lambda_1, \dots, \lambda_{2^n-1} \right]`
        q: Input qubits.
    """
    
    arg = ' '.join(str(i) for i in diag)
    plugin('ket_diag', q, arg)
    
def matrix(u00 : complex, u01 : complex, u10 : complex, u11 :complex, q : Optional[quant] = None):
    mat = [u00, u01, u10, u11]
    arg = ' '.join(str(i.real) for i in mat) +' '+' '.join(str(i.imag) for i in mat)
    if q is None:
        return lambda q : plugin('ket_mat', q, arg)
    else:
        plugin('ket_mat', q, arg)

def make_quantum(*args, **kwargs):
    r'''Make a Python function operate with quant variables.

    :Usage:

    .. code-block:: ket

        @make_quantum 
        def pow7mod15(reg1, reg2):
            reg2 = pow(7, reg1, 15)
            return reg1, reg2

        reg1, reg2 = quant(4), quant(4)
        pow7mod15(reg1, reg2)
    
    .. code-block:: ket

        balanced_oracle = make_quantum(func="""
            def oracle(a, b):
                def f(a):
                    return a
                return a, f(a)^b
        """, name='oracle')
        a, b = quant(2)
        balanced_oracle(a, b)
    
    '''
    
    if len(args) != 0 and callable(args[0]):
        func_str = '\n'.join(getsource(args[0]).split('\n')[1:])
        func_name = args[0].__name__
    else:
        func_str = kwargs['func']
        func_name = kwargs['name']

    func_b64 = b64encode(dedent(func_str).encode()).decode()
    def inner(*args):
        plugin_args = str(len(args)) + ' '
        for q in args:
            plugin_args += str(len(q)) + ' '
        plugin_args += func_name + ' ' + func_b64

        qubits = args[0]
        for q in args[1:]:
            qubits |= q
        plugin('ket_pycall', qubits, plugin_args)
    
    return inner

