<!--
SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>

SPDX-License-Identifier: Apache-2.0
-->

[![PyPI](https://img.shields.io/pypi/v/ket-lang.svg)](https://pypi.org/project/ket-lang/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![REUSE status](https://api.reuse.software/badge/gitlab.com/quantum-ket/ket)](https://api.reuse.software/info/gitlab.com/quantum-ket/ket)

# Ket Quantum Programming

[[_TOC_]]

Ket is an embedded programming language that introduces the ease of Python to quantum programming, letting anyone quickly prototype and test a quantum application.

Python is the most widely used programming language for machine learning and data science and has been the language of choice for quantum programming. Ket is a Python-embedded language, but many can use it as a Python library in most cases. So you can use Ket together with NumPy, ScyPy, Pandas, and other popular Python libraries.

Ket's goal is to streamline the development of hardware-independent classical quantum applications by providing transparent interaction of classical and quantum data. See <https://quantumket.org> to learn more about Ket.

## Installation :arrow_down:

Ket requires Python 3.8 or newer and is available for Linux, Windows, and macOS. If you are not using x86_64 (example ARM), you must install [Rust](https://www.rust-lang.org/tools/install) before installing Ket.

You can install Ket using [`pip`](https://pip.pypa.io/en/stable/user_guide/). To do so, copy and paste the following command into your terminal:

```shell
pip install ket-lang
```

## Documentation :scroll:

Documentation available at <https://quantumket.org>.

## Examples :bulb:

### Grover's Algorithm

```py
from ket import *
from math import sqrt, pi


def grover(n: int, oracle) -> int:
    p = Process()
    qubits = H(p.alloc(n))
    steps = int((pi / 4) * sqrt(2**n))
    for _ in range(steps):
        oracle(qubits)
        with around(H, qubits):
            phase_oracle(0, qubits)
    return measure(qubits).value


n = 8
looking_for = 13
print(grover(n, phase_oracle(looking_for)))
# 13
```

### Quantum Teleportation

```py
from ket import *


def entangle(a: Quant, b: Quant):
    return CNOT(H(a), b)


def teleport(quantum_message: Quant, entangled_qubit: Quant):
    adj(entangle)(quantum_message, entangled_qubit)
    return measure(entangled_qubit).value, measure(quantum_message).value


def decode(classical_message: tuple[int, int], qubit: Quant):
    if classical_message[0] == 1:
        X(qubit)

    if classical_message[1] == 1:
        Z(qubit)


if __name__ == "__main__":
    from math import pi

    p = Process()

    alice_message = PHASE(pi / 4, H(p.alloc()))

    alice_message_dump = dump(alice_message)

    alice_qubit, bob_qubit = entangle(*p.alloc(2))

    classical_message = teleport(
        quantum_message=alice_message, entangled_qubit=alice_qubit
    )

    decode(classical_message, bob_qubit)

    bob_qubit_dump = dump(bob_qubit)

    print("Alice Message:")
    print(alice_message_dump.show())

    print("Bob Qubit:")
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

## Setup for Ket Development :hammer:

To get started with Ket development, follow these steps:

1. **Rust Installation**
   
    Ensure that Rust is installed on your system. If not, follow the [Rust install guide](https://www.rust-lang.org/tools/install). After installation, set the Rust version to 1.75 using the following command:
    
    ```shell
    rustup default 1.82
    ```

2. **Clone and Compile**

    Clone the Ket repository and compile the Rust libraries:

    ```shell
    git clone --recursive https://gitlab.com/quantum-ket/ket.git
    cd ket

    cargo build --manifest-path src/ket/clib/libs/libket/Cargo.toml
    cargo build --manifest-path src/ket/clib/libs/kbw/Cargo.toml

    ln -s libket/target/debug/libket.so src/ket/clib/libs
    ln -s kbw/target/debug/libkbw.so src/ket/clib/libs
    ```

3. **Set Up Virtual Environment**

    Set up a virtual environment for Python:

    ```shell
    python3 -m venv venv
    source venv/bin/activate
    ```

4. **Install Dependencies**

    Upgrade pip and install development requirements:
    
    ```shell
    pip install -U pip
    pip install -r requirements_dev.txt
    ```

5. **Install Ket**

    Install Ket in editable mode:

    ```shell
    pip install -e .[full]
    ```

6. **Run Tests**

    To ensure everything is correctly installed, run the tests:
    
    ```shell
    pytest
    ```

You're now set up for Ket development! Happy coding! 🚀

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
