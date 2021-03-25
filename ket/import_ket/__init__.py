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

from ..preprocessor import ketpp
from ast import fix_missing_locations, parse
from os import path, PathLike
from types import ModuleType
from typing import Tuple

__all__ = ['import_ket', 'from_import_ket']

def __import_ket__(source : PathLike):
    with open(source, 'r') as file:
        tree = parse(file.read(), source)
    
    pp = ketpp()

    pp.visit(tree)
    fix_missing_locations(tree)

    return compile(tree, source, 'exec', optimize=2)
    
def __import_globals_ket__(source : PathLike, globals):
    exec(__import_ket__(source), globals)    

def import_ket(source : PathLike) -> ModuleType:
    """Import Ket file.

    Import a Ket source file as a Python module.

    :param source: Ket source file.  
    """

    _, file_name = path.split(source)
    module = ModuleType(file_name[:-4])

    src = 'from ket import *\nfrom ket.ket import label, branch, jump\n'
    obj = compile(src, '<ket build-in functions>', 'exec', optimize=2)
    exec(obj, module.__dict__)

    __import_globals_ket__(source, module.__dict__)
    return module

def from_import_ket(source : PathLike, *names) -> Tuple:
    """Import names from Ket file.
     
    :param source: Ket source file.  
    """
    
    module = import_ket(source)
    return tuple(module.__dict__[name] for name in names)
