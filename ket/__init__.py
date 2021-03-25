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

from .preprocessor import ketpp
from inspect import getsource
from ast import parse, fix_missing_locations
from typing import Callable

from .gates import *
from .standard import *
from .types import *
from .import_ket import *
from .gates import __all__ as all_gate
from .standard import __all__ as all_standard
from .types import __all__ as all_types
from .import_ket import __all__ as all_import

__all__ = all_gate+all_standard+all_types+all_import

def code_ket(func : Callable) -> Callable:
    """Parse as Ket function.
    
    Decorator to parse a Python function as a Ket Function.
    
    Worning:
        Do not use this decorator in a .ket file.    

    **Usage:**

    .. code-block:: python
        
        @code_ket
        def func(a : quant, b : quant, c : quant):
            m0 = measure(a)
            m1 = measure(b)
            if m1 == 1: # need @code_ket to properly execute
                x(c)
            if m0 == 1: # need @code_ket to properly execute
                z(c)
    """

    buildins = 'from ket import *\nfrom ket.ket import label, branch, jump\n'
    buildins_obj = compile(buildins, '<ket build-in functions>', 'exec', optimize=2)
    exec(buildins_obj, func.__globals__)
    
    tree = parse(getsource(func), '<function ' + func.__name__ + '>')
    pp = ketpp()

    pp.visit(tree)
    fix_missing_locations(tree)

    obj =  compile(tree, '<code_ket function ' + func.__name__ + '>', 'exec', optimize=2)
    exec(obj, func.__globals__)

    return func.__globals__[func.__name__]  
