from ket import *
from ket.amazon import AmazonBraket

amz = AmazonBraket(num_qubits=2)

p = Process()
q = p.alloc(2)

H(q[0])
CNOT(q[0], q[-1])

