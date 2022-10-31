
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

from .base import set_quantum_execution_target, set_process_features
from .clib.kbw import run_and_set_result, set_sim_mode_dense, set_sim_mode_sparse


def use_sparse():
    """Set KBW Sparse as quantum execution target"""

    set_sim_mode_sparse()
    set_process_features(plugins=['pown'])
    set_quantum_execution_target(run_and_set_result)


def use_dense():
    """Set KBW Dense as quantum execution target"""

    set_sim_mode_dense()
    set_process_features(plugins=['pown'])
    set_quantum_execution_target(run_and_set_result)
