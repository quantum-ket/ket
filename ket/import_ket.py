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

from .ketpp import ketpp
from types import ModuleType
from os import path
import ast

def __import_ket__(source : str, workdir : str):
    with open(path.join(workdir, source), 'r') as file:
        tree = ast.parse(file.read())
    
    pp = ketpp(workdir)

    pp.visit(tree)
    ast.fix_missing_locations(tree)

    return compile(tree, source, 'exec', optimize=2)
    
def __import_globals_ket__(source : str, workdir : str, globals):
    exec(__import_ket__(source, workdir), globals)    

def __import_module_ket__(source : str, workdir : str):
    module = ModuleType(source[:-4])

    src = '''from ket import *
from ket.import_ket import __import_module_ket__, __import_from_ket__, __import_globals_ket__

    '''
    obj = compile(src, 'ket build-in functions', 'exec', optimize=2)
    exec(obj, module.__dict__)

    __import_globals_ket__(source, workdir, module.__dict__)
    return module

def __import_from_ket__(source : str, workdir : str, *names):
    module = __import_module_ket__(source, workdir)
    return tuple(module.__dict__[name] for name in names)

def import_ket(source : str):
    workdir = path.dirname(path.abspath(source))
    return __import_module_ket__(path.basename(source), workdir)
