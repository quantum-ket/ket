<!--
SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>

SPDX-License-Identifier: Apache-2.0
-->

# Ket examples

Run any example with `python <file>`. Start from the top: each one adds a
concept.

| Example                                          | Shows                                                          |
| ------------------------------------------------ | -------------------------------------------------------------- |
| [`random_bits.py`](random_bits.py)               | `Process`, `alloc`, `H`, `measure`                             |
| [`bell.py`](bell.py)                             | entanglement, `sample`, observables with `obs` and `exp_value` |
| [`deutsch.py`](deutsch.py)                       | oracles as Python functions                                    |
| [`teleport.py`](teleport.py)                     | live execution: branching on measurement results               |
| [`grover.py`](grover.py)                         | `around`, `cat`, `qulib.oracle`                                |
| [`phase_estimation.py`](phase_estimation.py)     | `control`, `adj`, `QFT`                                        |
| [`execute_qasm.py`](execute_qasm.py)             | running OpenQASM 2.0 code, `dump`                              |
| [`cnot_count.py`](cnot_count.py)                 | gate counts after decomposition                                |
| [`maxcut.py`](maxcut.py)                         | QAOA with a classical optimizer (needs `scipy`)                |
| [`falqon.py`](falqon.py)                         | feedback-based optimization (needs `ket-lang[plot]`)           |
| [`logs.py`](logs.py)                             | logging, and finding the cause of an error                     |
| [`grover_gpu.py`](grover_gpu.py)                 | CPU vs GPU simulators (needs a GPU and `ket-lang[plot]`)       |
| [`bell_ibm.py`](bell_ibm.py)                     | running on IBM Quantum (needs `ket-lang[ibm]`)                 |
| [`bell_amazon_braket.py`](bell_amazon_braket.py) | running on Amazon Braket (needs `ket-lang[amazon]`)            |
