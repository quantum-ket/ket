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
