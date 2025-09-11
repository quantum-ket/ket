# SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2025 Ruan Luiz Molgero Lopes <ruan.molgero@grad.ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

import ket
from ket.amazon import AmazonBraket

from math import sqrt, pi
from functools import partial
from typing import Callable
from random import randint


def grover(size: int, oracle: Callable, outcomes: int = 1) -> int:
    """Grover's search algorithm.

    Args:
        size (int): The number of qubits to use for the search.
        oracle (Callable): The oracle function to use for the search.
        outcomes (int): The number of outcomes to search for.

    Returns:
        int: The measured value from the search.

    """

    braket = AmazonBraket()
    p = ket.Process(execution_target=braket)

    s = ket.H(p.alloc(size))

    steps = int((pi / 4) * sqrt(2**size / outcomes))
    for _ in range(steps):
        oracle(s)
        with ket.around(ket.cat(ket.H, ket.X), s):
            ket.CZ(*s)

    return ket.measure(s).get()


def qft(qubits: ket.Quant, do_swap: bool = True):
    """
    Applies the Quantum Fourier Transform (QFT) to the given qubits.

    Args:
        qubits (quant): A quantum register containing the qubits to be transformed.
        invert (bool): Whether inverse the qubits order at the end of the QFT. Default is True.
    """
    if len(qubits) == 1:
        ket.H(qubits)
    else:
        *head, tail = qubits
        ket.H(tail)

        for i, ctrl_qubit in enumerate(reversed(head)):
            with ket.control(ctrl_qubit):
                r = i + 2
                ket.P(2 * pi / 2**r, tail)

        qft(head, do_swap=False)

    if do_swap:
        size = len(qubits)
        for i in range(size // 2):
            ket.SWAP(qubits[i], qubits[size - i - 1])


def oracle(phase: float, i: int, tgr):
    """Performs a phase shift operation on the target
    qubit based on a given phase and index value.

    Args:
        phase_ (float): The phase value to be applied.
        i (int): The index value.
        tgr (quant): The target qubit.
    """
    ket.P(2 * pi * phase * 2**i, tgr)


def phase_estimator(oracle_gate, precision: int) -> float:
    """Estimates the phase of a given oracle function to a certain precision
    using the phase estimation algorithm.

    Args:
        oracle (callable): A function that performs a phase shift operation on
            the target qubit based on a given phase and index value.
            It should take the form: `oracle(phase_: float, i: int, tgr: quant) -> None`
        precision (int): The number of bits of precision desired in the estimation.

    Returns:
        float: An float value representing the estimated phase.
    """

    braket = AmazonBraket()
    p = ket.Process(execution_target=braket)

    ctr = ket.H(p.alloc(precision))
    trg = ket.X(p.alloc())

    for i, c in enumerate(ctr):
        with ket.control(c):
            oracle_gate(i, trg)

    ket.adj(qft)(ctr)

    return ket.measure(reversed(ctr)).get() / 2**precision


def quantum_sum(a, b, size):
    braket = AmazonBraket()
    p = ket.Process(execution_target=braket)

    qa = p.alloc(size + 1)
    qb = p.alloc(size)

    ket.qulib.math.set_int(qa, a)
    ket.qulib.math.set_int(qb, b)

    ket.qulib.math.addi(qa, qb)

    return ket.measure(qa).get()


def test_grover():
    SIZE = 5
    NUM_EXECUTIONS = 3
    SUCCESS_THRESHOLD = 0.50

    looking_for = randint(0, pow(2, SIZE) - 1)
    results = []

    for _ in range(NUM_EXECUTIONS):
        results.append(grover(SIZE, ket.qulib.oracle.phase_oracle(looking_for)))

    success_count = results.count(looking_for)
    rate = success_count / NUM_EXECUTIONS
    assert rate >= SUCCESS_THRESHOLD


def test_phase_estimator():
    estimate_pi = partial(phase_estimator, partial(oracle, pi / 10))
    assert abs((estimate_pi(16) * 10) - pi) < 1e-3


def test_quantum_adder():
    SIZE = 5

    for _ in range(6):
        a = randint(0, pow(2, SIZE) - 1)
        b = randint(0, pow(2, SIZE - 1) - 1)

        expected_result = a + b

        result = quantum_sum(a, b, SIZE)

        assert expected_result == result


if __name__ == "__main__":
    test_grover()
    test_phase_estimator()
    test_quantum_adder()
    print("OK")
