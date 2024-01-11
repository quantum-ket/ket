# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Deutsch algorithm"""

import ket


def balanced_oracle(x, y):
    """Applies a balanced oracle to the given qubits.

    The balanced oracle flips the second qubit (y) if and only if the first qubit (x) is |1⟩.
    U|x⟩|y⟩ = |x⟩|f(x) ⊕ y⟩, where f(x) = x

    Args:
        x: The first qubit.
        y: The second qubit.
    """
    ket.ctrl(x, ket.X)(y)


def constant_oracle(x, y):
    """Applies a constant oracle to the given qubits.

    The constant oracle flips the second qubit (y) regardless of the state of the first qubit (x).
    U|x⟩|y⟩ = |x⟩|f(x) ⊕ y⟩, where f(x) = 1

    Args:
        x: The first qubit.
        y: The second qubit.
    """
    ket.X(y)


def deutsch(oracle):
    """Runs the Deutsch algorithm with the given oracle.

    The Deutsch algorithm determines whether the given oracle is constant or balanced.
    Args:
        oracle: The oracle to be used in the algorithm.

    Returns:
        The measurement of the first qubit after the algorithm has been run.
    """
    p = ket.Process()
    x, y = p.alloc(2)  # xy = |00⟩
    ket.H(x)  # x = |+⟩
    ket.H(ket.X(y))  # y = |–⟩
    oracle(x, y)  # |–⟩|–⟩ if balanced else –|+⟩|–⟩
    return ket.measure(ket.H(x))


if __name__ == "__main__":
    # Run the Deutsch algorithm with the balanced and constant oracles.
    # Print the measurement of the first qubit after the algorithm has been run.
    print("Balanced oracle measurement:", deutsch(balanced_oracle).value)
    print("Constant oracle measurement:", deutsch(constant_oracle).value)
