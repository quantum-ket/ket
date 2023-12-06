from __future__ import annotations

# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

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
