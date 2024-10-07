from functools import partial
from ket import Process, H, RZZ, RX, Pauli, exp_value, sample

Y, Z = partial(Pauli, "Y"), partial(Pauli, "Z")


edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
n = 4

delta_t = 0.1

beta = [0]

num_layers = 10

process = Process(num_qubits=n, simulator="dense", execution="live")

qubits = H(process.alloc(n))
edges = list(map(qubits.at, edges))

for _ in range(num_layers):
    for a, b in edges:
        RZZ(delta_t, a, b)
    for q in qubits:
        RX(-beta[-1] * delta_t, q)

    beta.append(
        exp_value(
            sum(Y(a) * Z(b) for a, b in edges) + sum(Z(a) * Y(b) for a, b in edges)
        ).get()
    )


result = sample(qubits)

print(beta)
result.histogram().write_image("falqon.svg")
