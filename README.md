[![PyPI](https://img.shields.io/pypi/v/ket-lang.svg)](https://pypi.org/project/ket-lang/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

# Ket Quantum Programming

[[_TOC_]]

Ket is an embedded programming language that introduces the ease of Python to quantum programming, letting anyone quickly prototype and test a quantum application.

Python is the most widely used programming language for machine learning and data science and has been the language of choice for quantum programming. Ket is a Python-embedded language, but many can use it as a Python library in most cases. So you can use Ket together with NumPy, ScyPy, Pandas, and other popular Python libraries.


Ket's goal is to streamline the development of hardware-independent classical quantum applications by providing transparent interaction of classical and quantum data. See https://quantumket.org to learn more about Ket.

## Installation :arrow_down:

Ket requires Python 3.7 or newer and is available for Linux, Windows, and macOS. If you are not using x86_64 (example ARM), you must install [Rust](https://www.rust-lang.org/tools/install) before installing Ket.

You can install Ket using [`pip`](https://pip.pypa.io/en/stable/user_guide/). To do so, copy and paste the following command into your terminal:

```shell
pip install ket-lang
```

## Documentation :scroll:

Documentation available at https://quantumket.org.

## Examples :bulb:

### Grover's Algorithm

```py
from ket import *
from math import sqrt, pi


def grover(n: int, oracle) -> int:
    qubits = H(quant(n))
    steps = int((pi/4)*sqrt(2**n))
    for _ in range(steps):
        oracle(qubits)
        with around(H, qubits):
            phase_on(0, qubits)
    return measure(qubits).value


n = 8
looking_for = 13
print(grover(n, phase_on(looking_for)))
# 13
```

### Shor's Algorithm 

```py
from ket import *
from ket.lib import qft
from ket.plugins import pown
from random import randint
from functools import reduce
from math import log2, gcd


def quantum_subroutine(N, x):
    n = N.bit_length()

    def subroutine():
        reg1 = H(quant(n))
        reg2 = pown(x, reg1, N)
        measure(reg2)
        adj(qft, reg1)
        return measure(reg1).value

    r = reduce(gcd, [subroutine() for _ in range(n)])
    return 2**n//r


def shor(N: int) -> int:
    if N % 2 == 0:
        return 2
    n = N.bit_length()
    y = int(log2(N))
    for b in range(2, n+1):
        x = y/b
        u1 = int(2**x)
        u2 = u1+1
        if u1**b == N:
            return u1
        elif u2**b == N:
            return u2

    for _ in range(N.bit_length()):
        x = randint(2, N-1)
        gcd_x_N = gcd(x, N)
        if gcd_x_N > 1:
            return gcd_x_N

        r = quantum_subroutine(N, x)

        if r % 2 == 0 and pow(x, r//2) != -1 % N:
            p = gcd(x**(r//2)-1, N)
            if p != 1 and p != N and p*N//p == N:
                factor = p
                break

            q = gcd(x**(r//2)+1, N)
            if q != 1 and q != N and q*N//q == N:
                factor = q
                break

    if factor is not None:
        return factor
    else:
        raise RuntimeError(f"fails to factor {N}")

N = 4063
p = shor(N)
q = N//p
print(f'{N}={p}x{q}')
# 4063=17x239
```

### Quantum Teleportation

```py
from ket import *
from ket import code_ket


def entangle(a: quant, b: quant):
    return cnot(H(a), b)


def teleport(quantum_message: quant, entangled_qubit: quant):
    adj(entangle, quantum_message, entangled_qubit)
    return measure(entangled_qubit), measure(quantum_message)


@code_ket
def decode(classical_message: tuple[int, int], qubit: quant):
    if classical_message[0] == 1:
        X(qubit)

    if classical_message[1] == 1:
        Z(qubit)


if __name__ == '__main__':
    from math import pi

    alice_message = phase(pi/4, H(quant()))

    alice_message_dump = dump(alice_message)

    alice_qubit, bob_qubit = entangle(*quant(2))

    classical_message = teleport(
        quantum_message=alice_message,
        entangled_qubit=alice_qubit
    )

    decode(classical_message, bob_qubit)

    bob_qubit_dump = dump(bob_qubit)

    print('Alice Message:')
    print(alice_message_dump.show())

    print('Bob Qubit:')
    print(bob_qubit_dump.show())
# Alice Message:
# |0⟩     (50.00%)
#  0.707107               ≅      1/√2
# |1⟩     (50.00%)
#  0.500000+0.500000i     ≅  (1+i)/√4
# Bob Qubit:
# |0⟩     (50.00%)
#  0.707107               ≅      1/√2
# |1⟩     (50.00%)
#  0.500000+0.500000i     ≅  (1+i)/√4
```

## Ket Development :hammer:

Setup for Ket development:

```shell
git clone https://gitlab.com/quantum-ket/ket.git
cd ket
python setup.py build
```

If you are using [VS Code](https://code.visualstudio.com/), Ket has a [Dev Container](https://code.visualstudio.com/docs/remote/containers) :whale:.

## Roadmap :notebook_with_decorative_cover:

* [ ] Sample shots from dump variable.
* [ ] Create dump from quantum execution shots.
* [ ] Allow disabling some Ket features to verify quantum execution restriction at the classical runtime.
  * [ ] Use qubit after measurement.
  * [ ] Dump in the middle of the execution.
  * [ ] Classical control flow and binary operations.
* [ ] Quantum gate decomposition.
* [ ] Quantum code optimization.
* [ ] Quantum circuit visualization.
  
* :zap: We plan to expand the [quantum library](https://quantumket.org/ket#quantum-library) with quantum algorithm building blocks like the [`qft`](https://quantumket.org/ket#ket.lib.qft).  
* :package: Full quantum algorithm implementations must be packaged with  Ket as a dependency.
* :x: Low-level quantum control, like pulse programming, is out of Ket's scope.

## Cite Ket :book:

When using Ket for research projects, please cite:

> Evandro Chagas Ribeiro da Rosa and Rafael de Santiago. 2021. Ket Quantum Programming. J. Emerg. Technol. Comput. Syst. 18, 1, Article 12 (January 2022), 25 pages. DOI: [10.1145/3474224](https://doi.org/10.1145/3474224)

```bibtex
@article{ket,
   author = {Evandro Chagas Ribeiro da Rosa and Rafael de Santiago},
   title = {Ket Quantum Programming},
   year = {2021},
   issue_date = {January 2022},
   publisher = {Association for Computing Machinery},
   address = {New York, NY, USA},
   volume = {18},
   number = {1},
   issn = {1550-4832},
   url = {https://doi.org/10.1145/3474224},
   doi = {10.1145/3474224},
   journal = {J. Emerg. Technol. Comput. Syst.},
   month = oct,
   articleno = {12},
   numpages = {25},
   keywords = {Quantum programming, cloud quantum computation, qubit simulation}
}
```