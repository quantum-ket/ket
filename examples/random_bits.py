# SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

"""Quantum RNG."""

import ket


def random(n_bits: int) -> int:
    """Generates a random n_bits-bit number using a quantum circuit.

    Args:
        n_bits (int): Number of bits for the random number.

    Returns:
        int: Random n_bits-bit number.
    """
    p = ket.Process(simulator="dense", num_qubits=n_bits)
    return ket.measure(ket.H(p.alloc(n_bits))).value


if __name__ == "__main__":
    # Example usage
    N_BITS = 16
    print(N_BITS, "bits random number:", random(N_BITS))
