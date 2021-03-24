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

from ..ket import x, y, z, h, s, sd, t, td, u1, u2, u3, rx, ry, rz, quant

__all__ = ['i', 'x', 'y', 'z', 'h', 's', 'sd', 't', 'td', 'u1', 'u2', 'u3', 'rx', 'ry', 'rz']

def i(q : quant):
    r"""
     Identity gate 

    Apply an identity gate on every qubit of q.

    Note:
        This quantum gate does nothing! 

    **Matrix representation:**

    .. math::

        I = \begin{bmatrix} 
                     1 & 0 \\
                     0 & 1 
                 \end{bmatrix}

    **Effect:**

    .. math::

        \begin{matrix} 
                 X\left|0\right> = & \left|0\right> \\
                 X\left|1\right> = & \left|1\right>
             \end{matrix}

    :type q: :py:class:`quant`
    :param q: input qubits
    """
    return
 
    