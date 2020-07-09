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

def __import_module_ket__(source : str, workdir : str):
    module = ModuleType(source[:-4])

    obj = compile('from ket import *', 'ket build in functions', 'exec', optimize=2)
    exec(obj, module.__dict__)

    module.__dict__['__import_module_ket__'] = globals()['__import_module_ket__']

    exec(__import_ket__(source, workdir), module.__dict__)
    return module
