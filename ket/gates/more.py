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

from . import *
from ..ket import quant
from math import pi

def u2(phi : float, _lambda : float, q : quant):
    r"""Generic rotation gate with 2 Euler angles

    Apply a rotation about the X+Z axis on every qubit of q.

    **Matrix representation:**

     .. math::

         U2(\phi, \lambda) =
                  \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & -e^{i\lambda} \\
                                            e^{i\phi} & e^{i(\lambda+\phi)} \end{bmatrix}

    **Effect:**

    .. math::

        \begin{matrix} 
                 U2\left|0\right> = & \frac{1}{\sqrt{2}}\left(\left|0\right> - e^{i\lambda}\left|1\right>\right) \\
                 U2\left|1\right> = & \frac{1}{\sqrt{2}}\left(e^{i\phi}\left|0\right> + e^{i(\lambda+\phi)}\left|1\right>\right) 
             \end{matrix}

    :param q: input qubits
    :param phi: :math:`\phi`
    :param _lambda: :math:`\lambda`
    """     
    RZ(_lambda, q)    
    RY(pi/2, q)
    RZ(phi, q)

def u3(theta : float, phi : float, _lambda : float, q : quant):
    r"""Generic rotation gate with 3 Euler angles

    Apply a generic rotation on every qubit of q.

    **Matrix representation:**

     .. math::

         U3(\theta, \phi, \lambda) = \begin{bmatrix}
                       \cos{\frac{\theta}{2}} & -e^{i\lambda}\sin{\frac{\theta}{2}} \\
              e^{i\phi}\sin{\frac{\theta}{2}} & e^{i(\lambda+\phi)}\cos{\frac{\theta}{2}}  
                                          \end{bmatrix}

    **Effect:**

    .. math::

        \begin{matrix} 
                 U3\left|0\right> = & \cos{\theta\over2}\left|0\right> -e^{i\lambda}\sin{\theta\over2}\left|1\right> \\
                 U3\left|1\right> = & e^{i\phi}\sin{\theta\over2}\left|0\right> + e^{i(\lambda+\phi)}\cos{\theta\over2}\left|1\right> 
             \end{matrix}

    :param q: input qubits
    :param theta: :math:`\theta`
    :param phi: :math:`\phi`
    :param lambda: :math:`\lambda`
    """

    RZ(_lambda, q)
    RX(pi/2, q)
    RZ(theta, q)
    RX(-pi/2, q)
    RZ(phi, q)

i  = I            
x  = X 
y  = Y
z  = Z
h  = H
s  = S
sd = SD
t  = T
td = TD
p = phase
P = p
u1 = p
U1 = p
U2 = u2
U = u3
U3 = u3
rx = RX
ry = RY
rz = RZ
