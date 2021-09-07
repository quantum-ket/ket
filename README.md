[![PyPI](https://img.shields.io/pypi/v/ket-lang.svg)](https://pypi.org/project/ket-lang/)
[![AppImage](https://gitlab.com/quantum-ket/ket/badges/master/pipeline.svg)](https://gitlab.com/quantum-ket/ket/-/jobs)

# Ket Quantum Programming Language

Ket is a Python-embedded  language for hybridity classical-quantum  programming.

### Table of contents:

* [Code examples](#code-examples)
* [Usage](#usage)
* [Installation](#installation)

## Code examples

### Random Number Generation

```python
# random.ket
def random(n_bits):
  with run():
     q = quant(n_bits)
     H(q)
     return measure(q).get()

n_bits = 32
print(n_bits, 'bits random number:', random(n_bits))
```

```console
$ ket random.ket
32 bits random number: 3830764503
```

### Quantum Teleportation:

```python
# teleport.ket
def teleport(alice : quant) -> quant:
    alice_b, bob_b = quant(2)
    ctrl(H(alice_b), X, bob_b)

    ctrl(alice, X, alice_b)
    H(alice)

    m0 = measure(alice)
    m1 = measure(alice_b)

    if m1 == 1:
        X(bob_b)
    if m0 == 1:
        Z(bob_b)

    return bob_b

alice = quant(1)         # alice = |0⟩
H(alice)                 # alice = |+⟩
Z(alice)                 # alice = |–⟩
bob = teleport(alice)    # bob  <- alice
H(bob)                   # bob   = |1⟩
bob_m = measure(bob)

print('Expected measure 1, result =', bob_m.get())
```

```console
$ ket teleport.ket
Expected measure 1, result = 1
```

## Usage 

```console
$ ket -h
Ket program options:
  -h [ --help ]              Show this information.
  -o [ --out ]               KQASM output file.
  -s [ --kbw ]  (=127.0.0.1) Quantum execution (KBW) address.
  -p [ --port ]  (=4242)     Quantum execution (KBW) port.
  --seed                     Set RNG seed for quantum execution.
  --api-args                 Additional parameters for quantum execution.
  --no-execute               Does not execute KQASM, measurements return 0.
  --dump-to-fs               Use the filesystem to transfer dump data.
```

## Installation

> Ket Bitwise Simulator is required for quantum execution. See
> https://gitlab.com/quantum-ket/kbw#installation for installation instructions.

Available installation methods:

* [pip](#install-using-pip)
* [Source](#install-from-source)

### Install using pip

Installing from PyPI:

```console
$ pip install ket-lang
```

Installing the last version from git:

```console
$ pip install git+https://gitlab.com/quantum-ket/ket.git
```

> Compiled manylinux wheel available [here](https://gitlab.com/quantum-ket/ket/-/jobs/artifacts/master/download?job=python_manylinux_wheel)

### Install from source 

Install requirements:

* C/C++ compiler
* CMake
* Ninja or GNU Make
* Conan

To install from source runs:

```console
$ git clone --recurse-submodules https://gitlab.com/quantum-ket/ket.git
$ cd ket
$ python setup.py install
```

-----------

This project is part of the Ket Quantum Programming, see the documentation for
more information https://quantum-ket.gitlab.io.
