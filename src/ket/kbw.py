# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

from .base import set_quantum_execution_target, set_process_features
from .clib.kbw import (run_and_set_result, set_sim_mode_dense,
                       set_sim_mode_sparse, set_seed, set_dump_type)

__all__ = ['use_sparse', 'use_dense', 'set_seed', 'set_dump_type']


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
