
from .base import set_quantum_execution_target
from .clib.kbw import run_and_set_result, set_sim_mode_dense, set_sim_mode_sparse


def use_sparse():
    set_sim_mode_sparse()
    set_quantum_execution_target(run_and_set_result)


def use_dense():
    set_sim_mode_dense()
    set_quantum_execution_target(run_and_set_result)
