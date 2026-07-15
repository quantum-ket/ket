<!--
SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>

SPDX-License-Identifier: Apache-2.0
-->

# Ket Quantum Programming

[![PyPI](https://img.shields.io/pypi/v/ket-lang.svg)](https://pypi.org/project/ket-lang/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![REUSE status](https://api.reuse.software/badge/gitlab.com/quantum-ket/ket)](https://api.reuse.software/info/gitlab.com/quantum-ket/ket)

Ket is an embedded programming language that introduces the ease of Python to quantum programming, letting anyone quickly prototype and test a quantum application.

Python is the most widely used programming language for machine learning and data science and has been the language of choice for quantum programming. Ket is a Python-embedded language, but many can use it as a Python library in most cases. So you can use Ket together with NumPy, SciPy, Pandas, and other popular Python libraries.

Ket's goal is to streamline the development of hardware-independent classical quantum applications by providing transparent interaction of classical and quantum data. See <https://quantumket.org> to learn more about Ket.

## Installation ⬇️

Ket requires Python 3.10 or newer and is available for Linux, Windows, and macOS (both Apple silicon and Intel). If you are using a non-x86_64 (Intel/AMD) CPU, such as ARM, on Linux or Windows, you will need to install [Rust](https://www.rust-lang.org/tools/install) before installing Ket.

You can install Ket using [`pip`](https://pip.pypa.io/en/stable/user_guide/). To do so, copy and paste the following command into your terminal:

```shell
pip install ket-lang
```

## Documentation 📜

Documentation available at <https://quantumket.org>.

## Examples 💡

### Grover's Algorithm

```python
import ket
from math import sqrt, pi


def grover(size: int, oracle) -> int:
    p = ket.Process(simulator="dense", num_qubits=size)
    s = ket.H(p.alloc(size))
    steps = int((pi / 4) * sqrt(2**size))
    for _ in range(steps):
        oracle(s)
        with ket.around(ket.cat(ket.H, ket.X), s):
            ket.CZ(*s)
    return ket.measure(s).value


size = 8
looking_for = 13
print(grover(size, ket.qulib.oracle.phase_oracle(looking_for)))
# 13
```

### Quantum Teleportation

```python
import ket


def bell(qubits):
    """Prepare a Bell (maximally entangled) pair."""
    return ket.ctrl(ket.H(qubits[0]), ket.X)(qubits[1])


def teleport(alice_msg, alice_aux, bob_aux):
    """Teleport the quantum state of alice_msg to bob_aux."""
    ket.ctrl(alice_msg, ket.X)(alice_aux)
    ket.H(alice_msg)

    m0 = ket.measure(alice_msg)
    m1 = ket.measure(alice_aux)

    if m1.value == 1:
        ket.X(bob_aux)
    if m0.value == 1:
        ket.Z(bob_aux)

    return bob_aux


p = ket.Process(execution="live")

alice = p.alloc()   # alice = |0⟩
ket.H(alice)        # alice = |+⟩
ket.Z(alice)        # alice = |–⟩

bob = teleport(alice, *bell(p.alloc(2)))  # bob ← alice
ket.H(bob)          # bob   = |1⟩
print("Expected measure 1, result =", ket.measure(bob).value)
# Expected measure 1, result = 1
```

## Setup for Ket Development 🔨

To get started with Ket development, follow these steps:

1. **Rust Installation**

   Ensure that Rust is installed on your system. If not, follow the [Rust install guide](https://www.rust-lang.org/tools/install).

   **Important**: For Rust development, make sure to use the exact same rustc version as the current ket distribution. This version-matching is not required if you are developing with Python or C FFI.

2. **Clone and Compile**

   Clone the Ket repository and compile the Rust libraries:

   ```shell
   git clone --recursive https://gitlab.com/quantum-ket/ket.git
   cd ket

   cargo build --manifest-path src/ket/clib/libs/libket/Cargo.toml
   cargo build --manifest-path src/ket/clib/libs/kbw/Cargo.toml

   # Create symlinks from within the libs directory
   ln -s libket/target/debug/libket.so src/ket/clib/libs/libket.so
   ln -s kbw/target/debug/libkbw.so src/ket/clib/libs/libkbw.so
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
   pip install -e .
   ```

6. **Run Tests**

   To ensure everything is correctly installed, run the tests:

   ```shell
   pytest
   ```

You're now set up for Ket development! Happy coding! 🚀

## Cite Ket 📖

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
