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
