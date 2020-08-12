from .. import *
from math import pi

def qft(q : quant):
    """Apply a Quantum Fourier Transformation.
    
    note:
        The operation leaves the qubits in the inverse order.
    """
    
    lambd = lambda k : pi*k/2
    for i in range(len(q)):
        for j in range(i):
            ctrl(q[i], u1, lambd(i-j), q[j])
        h(q[i])

def cnot(c, t):
    """Quantum-bitwise Controlled-NOT."""
    for i, j in zip(c, t):
        ctrl(i, x, j)

def bell(aux0 : int, aux1 : int) -> quant:
    """Return two entangle qubits in the Bell state."""
    q = quant(2)
    if aux0 == 1:
        x(q[0])
    if aux1 == 1:
        x(q[1])
    h(q[0])
    ctrl(q[0], x, q[1])
    return q

