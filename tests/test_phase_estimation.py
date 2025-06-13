# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Phase estimation algorithm"""

from math import pi
from functools import partial
import ket


def qft(qubits: ket.Quant, invert: bool = True):
    """
    Applies the Quantum Fourier Transform (QFT) to the given qubits.

    Args:
        qubits (quant): A quantum register containing the qubits to be transformed.
        invert (bool): Whether inverse the qubits order at the end of the QFT. Default is True.
    """
    if len(qubits) == 1:
        # Base case:
        # single-qubit QFT is simply the Hadamard gate.
        ket.H(qubits)
    else:
        # Recursive case:
        # Apply a Hadamard gate to the last qubit.
        *head, tail = qubits
        ket.H(tail)

        # Apply controlled phase shift gates to the last qubit with each of the remaining qubits.
        for i, ctrl_qubit in enumerate(reversed(head)):
            with ket.control(ctrl_qubit):
                ket.P(pi / 2 ** (i + 1), tail)

        # Recursively apply QFT to the remaining qubits.
        qft(head, invert=False)

    # Swap the order of the qubits.
    if invert:
        size = len(qubits)
        for i in range(size // 2):
            ket.SWAP(qubits[i], qubits[size - i - 1])


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

    p = ket.Process(simulator="dense", num_qubits=precision + 1)

    # Create a quantum register to hold the control qubits.
    ctr = ket.H(p.alloc(precision))

    # Create a target qubit.
    tgr = ket.X(p.alloc())

    # Apply the oracle function with each control qubit.
    for i, c in enumerate(ctr):
        with ket.control(c):
            oracle_gate(i, tgr)

    # Apply the inverse QFT to the control qubits.
    ket.adj(qft)(ctr)

    # Measure the control qubits and return the result.
    return ket.measure(reversed(ctr)).value / 2**precision


def oracle(phase: float, i: int, tgr):
    """Performs a phase shift operation on the target
    qubit based on a given phase and index value.

    Args:
        phase_ (float): The phase value to be applied.
        i (int): The index value.
        tgr (quant): The target qubit.
    """
    ket.P(2 * pi * phase * 2**i, tgr)


estimate_pi = partial(phase_estimator, partial(oracle, pi / 10))


def test_phase_estimator():
    assert abs((estimate_pi(16) * 10) - pi) < 1e-3


if __name__ == "__main__":
    test_phase_estimator()
    print("OK")
