from inspect import getsource, getsourcefile
from .ketpp import ketpp
from ast import parse, fix_missing_locations
from types import ModuleType

def code_ket(func):
    tree = parse(getsource(func), '<function ' + func.__name__ + '>')
    pp = ketpp(getsourcefile(func))

    pp.visit(tree)

    fix_missing_locations(tree)

    obj =  compile(tree, '<code_ket function ' + func.__name__ + '>', 'exec', optimize=2)
    exec(obj, func.__globals__)

    return func.__globals__[func.__name__]  
