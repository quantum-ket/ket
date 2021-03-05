# MIT License
# 
# Copyright (c) 2020 Evandro Chagas Ribeiro da Rosa <evandro.crr@posgrad.ufsc.br>
# Copyright (c) 2020 Rafael de Santiago <r.santiago@ufsc.br>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from ..ket import plugin, quant
from typing import List
from inspect import getsource
from base64 import b64encode
from textwrap import dedent

def pown(a : int, x : quant, n : int) -> quant: 
    """Apply a modular exponentiation in a superposition.

    .. math::

        \left| x \right> \left| 0 \right> \rightarrow  \left| x \right> \left| a^x\; \text{mod} \, n \right>

    :return: Quant with the result of the operation.
    """
    
    ret = quant(n.bit_length())
    arg = str(len(x)) + ' ' + str(a) + ' ' + str(n)

    plugin('ket_pown', x|ret, arg)

    return ret

def diagonal(diag : List[float], q : quant):
    """Apply a diagonal matrix.

    .. math::

        \begin{bmatrix}
                 e^{i\lambda_0} & 0              & \cdots & 0 \\
                 0              & e^{i\lambda_1} & \cdots & 0 \\
                 \vdots         & \vdots         & \ddots & \vdots \\
                 0              & 0              & \cdots & e^{i\lambda_{2^n-1}} 
             \end{bmatrix}

    :param diag: :math:`\left[ \lambda_0, \lambda_1, \dots, \lambda_{2^n-1} \right]`
    """
    
    arg = ' '.join(str(i) for i in diag)
    plugin('ket_diag', q, arg)
    

def make_quantum(*args, **kwargs):
    """Make a Python function operate with quant variables.

    Usage:

    .. code-block:: ket

        @make_quantum 
        def pow7mod15(reg1, reg2):
            reg2 = pow(7, reg1, 15)
            return reg1, reg2

        reg1, reg2 = quant(4), quant(4)
        pow7mod15(reg1, reg2)
    
    .. code-block:: ket

        balanced_oracle = make_quantum(func=\"""
            def oracle(a, b):
                def f(a):
                    return a
                return a, f(a)^b
        \""", name='oracle')
        a, b = quant(2)
        balanced_oracle(a, b)
    
    """
    
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

