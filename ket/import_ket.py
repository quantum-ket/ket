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

from .preprocessor import ketpp
from inspect import getsource
from ast import fix_missing_locations, parse
from os import path, PathLike
from types import ModuleType
from typing import Callable

__all__ = ['import_ket', 'from_import_ket']


def _import_ket(source: PathLike):
    with open(source, 'r') as file:
        tree = parse(file.read(), source)

    pp = ketpp()

    pp.visit(tree)
    fix_missing_locations(tree)

    return compile(tree, source, 'exec', optimize=2)


def _import_globals_ket(source: PathLike, globals):
    exec(_import_ket(source), globals)


buildins = \
    """
from ket import *
from ket.base import label, branch, jump
from ket.preprocessor import *
"""


def import_ket(source: PathLike) -> ModuleType:
    """Import Ket file

    Import a Ket source file as a Python module.

    Args:
        source: Ket source file.  
    """

    _, file_name = path.split(source)
    module = ModuleType(file_name[:-4])

    obj = compile(buildins, '<ket build-in functions>', 'exec', optimize=2)
    exec(obj, module.__dict__)

    _import_globals_ket(source, module.__dict__)
    return module


def from_import_ket(source: PathLike, *names: list[str]) -> tuple:
    """Import names from Ket file

    Args:
        source: Ket source file.  
        names: Names to import from ``source``.
    """

    module = import_ket(source)
    return tuple(module.__dict__[name] for name in names)


def code_ket(func: Callable) -> Callable:
    """Parse as Ket function

    Decorator to parse a Python function as a Ket Function.

    Warning:
        * Do not use this decorator in a .ket file.
        * This decorator do not work with function defined inside a Jupyter Notebook or the Python interpreter. 
        * This decorator is not part of Ket's preamble.

    :Usage:

    .. code-block:: python

        from ket import code_ket

        @code_ket
        def func(a : quant, b : quant, c : quant):
            m0 = measure(a)
            m1 = measure(b)
            if m1 == 1: # need @code_ket to properly execute
                x(c)
            if m0 == 1: # need @code_ket to properly execute
                z(c)
    """

    if '__in_ket__' in globals() and globals()['__in_ket__']:
        return func

    in_ket = "__in_ket__{}__".format(func.__name__)
    if in_ket in func.__globals__ and func.__globals__[in_ket]:
        return func

    doc = func.__doc__
    annotations = func.__annotations__

    buildins_more = buildins+'\n'+in_ket+' = True\n'
    buildins_obj = compile(
        buildins_more, '<ket build-in functions>', 'exec', optimize=2)
    exec(buildins_obj, func.__globals__)

    tree = parse(getsource(func), '<function ' + func.__name__ + '>')
    pp = ketpp()

    pp.visit(tree)
    fix_missing_locations(tree)

    obj = compile(tree, '<code_ket function ' +
                  func.__name__ + '>', 'exec', optimize=2)
    exec(obj, func.__globals__)
    func.__globals__[func.__name__].__doc__ = doc
    func.__globals__[func.__name__].__annotations__ = annotations

    return func.__globals__[func.__name__]
