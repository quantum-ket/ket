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

from .base import plugin, quant, base_X

__all__ = ['plugin', 'pown']


def pown(a: int, x: quant, N: int) -> quant:  # pylint: disable=invalid-name
    r"""Apply a modular exponentiation in a superposition

    .. math::

        \left| x \right> \left| 1 \right> \rightarrow  \left| x \right> \left| a^x\; \text{mod} \, N \right>

    .. note::

        Plugin availability depends on the quantum execution target.

    Args:
        a: :math:`a`.
        x: :math:`x`.
        N: :math:`N`.

    :return: :class:`~ket.base.quant` with the operation result.
    """  # pylint: disable=line-too-long

    ret = quant(N.bit_length())
    base_X(ret[-1])

    plugin('pown', f'{a} {N}', x + ret)

    return ret
