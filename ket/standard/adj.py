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

import types
from ..ket import adj_begin, adj_end
from typing import Union, Callable, Iterable, Generator
from inspect import signature
from types import GeneratorType

class inverse:
    """Execute inverse

    Apply the quantum operations backwards.
    
    **Usage:**

    .. code-block:: ket

        with inverse():
            ...
            
    """
    def __enter__ (self):
        adj_begin()     

    def __exit__ (self, type, value, tb):
        adj_end()     

def adj(func : Union[Callable, Iterable[Callable]], *args, **kwargs):
    """Call the inverse of a quantum operation."""
    
    if len(signature(func).parameters) != 0 and len(args) == 0 and len(kwargs) == 0:
        def __adj__(*args, **kwargs):
            adj(func, *args, **kwargs)
        return __adj__

    ret = []
    adj_begin()
    if hasattr(func, '__iter__'):
        for f in func:
            ret.append(f(*args, **kwargs))
    else:
        ret = func(*args, **kwargs)
    adj_end()
    return ret

class around:
    def __init__(self, outter_func : Union[Callable, Iterable[Callable], Generator[Callable, None, None]], *args, **kwargs):
        if type(outter_func) == GeneratorType:
            self.outter_func = list(outter_func)
        else:
            self.outter_func = outter_func
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        if hasattr(self.outter_func, '__iter__'):
            for func in self.outter_func:
                func(*self.args, **self.kwargs)
        else:
            self.outter_func(*self.args, **self.kwargs)

    def __exit__ (self, type, value, tb):
        adj(self.__enter__)
