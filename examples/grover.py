from ket import *
from math import sqrt, pi
from typing import Callable


def grover(size: int, oracle: Callable, outcomes: int = 1) -> int:
    """Grover's search algorithm.

    Args:
        size (int): The number of qubits to use for the search.
        oracle (Callable): The oracle function to use for the search.
        outcomes (int): The number of outcomes to search for.

    Returns:
        int: The measured value from the search.

    """
    s = H(quant(size))  # Initialize the state to a superposition of all possible outcomes.

    steps = int((pi / 4) * sqrt(2**size / outcomes))  # Calculate the number of iterations for the algorithm.
    for _ in range(steps):
        oracle(s)  # Apply the oracle function to the state.
        with around(H, s):  # Apply a Hadamard gate to each qubit.
            phase_on(0, s)  # Apply a phase gate to each qubit.

    return measure(s).value  # Measure the final state and return the result.


if __name__ == '__main__':
    from ket import kbw
    from random import randint

    size = 12
    looking_for = randint(0, pow(2, size) - 1)

    print("Searching for value", looking_for, "using", size, "qubits.")

    kbw.use_dense()  # Use KBW Dense representation for quantum simulation.

    print('Dense Simulation: measured', grover(size, phase_on(looking_for)))
    print('Execution time:', quantum_exec_time())

    kbw.use_sparse()  # Use KBW Sparse representation for quantum simulation.
    print('Sparse Simulation: measured', grover(size, phase_on(looking_for)))
    print('Execution time:', quantum_exec_time())
