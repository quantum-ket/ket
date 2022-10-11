[![PyPI](https://img.shields.io/pypi/v/ket-lang.svg)](https://pypi.org/project/ket-lang/)
[![AppImage](https://gitlab.com/quantum-ket/ket/badges/master/pipeline.svg)](https://gitlab.com/quantum-ket/ket/-/jobs)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)


# Ket Quantum Programming Language

Ket is a Python-embedded language for hybridity classical-quantum  programming.

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
     return measure(q).value

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

print('Expected measure 1, result =', bob_m.value)
```

```console
$ ket teleport.ket
Expected measure 1, result = 1
```

## Usage 

```console
$ ket -h
usage: ket [-h] [--version] [-o OUT] [-s SEED] .ket

Ket interpreter

positional arguments:
  .ket                  source code

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
```

## Installation

Available installation methods:

* [pip](#installing-with-pip)
* [Source](#installing-from-source)

### Installing with pip

Installing from PyPI:

```console
$ pip install ket-lang
```
Installing latest Gitlab CI build:

```console
$ pip install "https://gitlab.com/quantum-ket/ket/-/jobs/artifacts/master/raw/wheelhouse/ket_lang-`wget -O- -q https://gitlab.com/quantum-ket/ket/-/raw/master/ket/__version__.py | awk -F\' '{print $2}'`-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl?job=wheelhouse"
```

### Installing from source 

Requirements:

* [Rust](https://www.rust-lang.org/tools/install)

To install from source runs:

```console
$ git clone https://gitlab.com/quantum-ket/ket.git
$ cd ket
$ ./util/make_libs.sh
$ python setup.py install
```

-----------

This project is part of the Ket Quantum Programming, see the documentation for
more information https://quantumket.org.
