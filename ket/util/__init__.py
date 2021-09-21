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

from ..ket import report, metrics, context, config, build_info as ket_version

__all__ = ['ket_config', 'report', 'context', 'ket_version']

def ket_config(**params):
    for param in params:
        if type(params[param]) == bool:
          params[param] = int(params[param])
        config(param, str(params[param]))

report.__doc__ = \
    """Return the current KQASM metrics 

    **Example:**

    .. code-block:: ket

        >>> q = quant(20)
        >>> ctrl(H(q[0]), X, q[1:])
        >>> metrics = report()
        >>> print(metrics)
        Qubits used:         	20
        Free qubits:         	0
        Allocated Qubits:    	20
          ↳ Max concurrently:	20
        Measurements:        	0
        Quantum gates:       	20
          ↳ X gate:         	19
          ↳ H gate:         	1
        Quantum CTRL gates:  	19
          ↳ 1 control:      	19
        Ket Bitwise plugins: 	0

    :rtype: :class:`ket.util.metrics`
    """

metrics.__repr__ = lambda self : '<Ket metrics; '+self.this.__repr__()+'>'