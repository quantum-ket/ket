from ket import *


def random(n_bits: int) -> int:
    """Generates a random n_bits-bit number using a quantum circuit.

    Args:
        n_bits (int): Number of bits for the random number.

    Returns:
        int: Random n_bits-bit number.
    """
    with run():
        return measure(H(quant(n_bits))).value


if __name__ == '__main__':
    # Example usage
    n_bits = 16
    print(n_bits, 'bits random number:', random(n_bits))
