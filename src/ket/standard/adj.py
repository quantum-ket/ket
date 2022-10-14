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

from typing import Callable, Any
from inspect import signature
from types import GeneratorType
from ..base import adj_begin, adj_end


class inverse:  # pylint: disable=invalid-name
    r"""Open a inverse scope

    Inside a ``with inverse`` scope, the the quantum operations backwards.

    :Usage:

    .. code-block:: ket

        with inverse():
            ...

    """

    def __enter__(self):
        adj_begin()

    def __exit__(self, type, value, tb):  # pylint: disable=redefined-builtin, invalid-name
        adj_end()


def _adj(func: Callable | list[Callable],
         *args,
         **kwargs) -> Any:
    """Call inverse operation"""

    adj_begin()
    if hasattr(func, '__iter__'):
        ret = []
        for gate in func:
            ret.append(gate(*args, **kwargs))
    else:
        ret = func(*args, **kwargs)
    adj_end()
    return ret


def adj(func: Callable | list[Callable],
        *args,
        later_call: bool = False,
        **kwargs) -> Callable | Any:
    """Inverse of a Callable

    :Call inverse:

    .. code-block:: ket

        ret1 = adj(func, *args, **kwargs)
        # Equivalent to:
        # with inverse():
        #     ret1 = func(*args, **kwargs)

        ret2 = adj([f0, f1, f2, ...], *args, **kwargs)
        # Equivalent to:
        # ret2 = []
        # with inverse():
        #     for f in func:
        #         ret2.append(f(*args, **kwargs))


    :Create inverse operation:

    1. If the keyword argument ``later_call`` is ``True``, return a
    ``Callable[[], Any]``:

    .. code-block:: ket

        adj_func = adj(func, *args, **kwargs, later_call=True)
        # Equivalent to:
        # adj_func = lambda : adj(func, *args, **kwargs)

    2. If any argument for ``func`` is provided, return the ``func`` inverse:

    .. code-block:: ket

        # def func(*args, *kwargs): ...
        adj_func = adj(func)
        # Equivalent to:
        # adj_func = lambda *args, **kwargs : adj(func, *args, **kwargs)

    Example:

    .. code-block:: ket

        rx_pi_inv = adj(RX(pi))

    Args:
        func: Function or list of functions to add control.
        args: ``func`` arguments.
        kwargs: ``func`` keyword arguments.
        later_call: If ``True``, do not execute and return a ``Callable[[], Any]``.
    """

    if len(signature(func).parameters) != 0 and len(args) == len(kwargs) == 0:
        return lambda *args, **kwargs: _adj(func, *args, **kwargs)
    if later_call:
        return lambda: _adj(func, *args, **kwargs)
    return _adj(func, *args, **kwargs)


class around:  # pylint: disable=invalid-name
    r"""Apply operation U around V and V inverse

    With the quantum operations U and V, execute VUV :math:`\!^\dagger`, where V
    is defined as by ``func, *args, **kwargs`` and U is the open scope.

    * ``func`` must be a ``Callable`` or ``Iterable[Callable]``.

    .. code-block:: ket

        with around(V):
            U

    :Example:

    .. code-block:: ket

        # Grover diffusion operator
        with around([H, X], s):
            with control(s[1:]):
                Z(s[0])

    Args:
        func: ``V`` operation.
        args: ``func`` arguments.
        kwargs: ``func`` keyword arguments.
    """

    def __init__(self,
                 func: Callable | list[Callable],
                 *args,
                 **kwargs):
        self.outer_func = list(func) if isinstance(
            func, GeneratorType) else func
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        if hasattr(self.outer_func, '__iter__'):
            for func in self.outer_func:
                func(*self.args, **self.kwargs)
        else:
            self.outer_func(*self.args, **self.kwargs)

    def __exit__(self, type, value, tb):  # pylint: disable=redefined-builtin, invalid-name
        adj(self.__enter__)
