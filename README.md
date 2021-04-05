[![ket](https://snapcraft.io//ket/badge.svg)](https://snapcraft.io/ket)
[![PyPI](https://img.shields.io/pypi/v/ket-lang.svg)](https://pypi.org/project/ket-lang/)
[![AppImage](https://gitlab.com/quantum-ket/ket/badges/master/pipeline.svg)](https://gitlab.com/quantum-ket/ket/-/jobs)

# Ket Quantum Programming Language

Ket is a Python-embedded classical-quantum programming language for dynamic
interaction between classical and quantum computers.

### Table of contents:

* [Code examples](#code-examples)
* [Usage](#usage)
* [Installation](#installation)
* [Run examples](#run-examples)


## Code examples

### Random Number Generation

```python
def random(n_bits):
  with run():
     q = quant(n_bits)
     h(q)
     return measure(q).get()

n_bits = 32
print(n_bits, 'bits random number:', random(n_bits))
```

> 32 bits random number: 3830764503

### Quantum Teleportation:

```python
from ket.lib import bell

def teleport(a):
    b = bell(0, 0)
    with control(a):
        x(b[0])
    h(a)
    m0 = measure(a)
    m1 = measure(b[0])
    if m1 == 1:
        x(b[1])
    if m0 == 1:
        z(b[1])
    return b[1]

a = quant(1)    # a = |0⟩
h(a)            # a = |+⟩ 
z(a)            # a = |-⟩
y = teleport(a) # y <- a
h(y)            # y = |1⟩
print('Expected measure 1, result =', measure(y).get())
```

> Expected measure 1, result = 1

### Shor's Algorithm Factoring 15:

```python
from math import pi, gcd
from ket import plugins
from ket.lib import qft

def period():
    reg1 = quant(4)
    h(reg1)
    reg2 = plugins.pown(7, reg1, 15)
    qft(reg1)
    return measure(reg1).get()

r = period()
results = [r]
for _ in range(4):
    results.append(period())
    r = gcd(r, results[-1])

print(results)
r = 2**4/r

print('measurements =', results)
print('r =', r)
p = gcd(int(7**(r/2))+1, 15)
q = gcd(int(7**(r/2))-1, 15)
print(15, '=', p , "x", q)
```

> [8, 12, 12, 4, 8]\
> measurements = [8, 12, 12, 4, 8]\
> r = 4.0\
> 15 = 5 x 3

## Usage 

```console
$ ket -h
Ket program options:
  -h [ --help ]              Show this information
  -o [ --out ]               KQASM output file
  -s [ --kbw ]  (=127.0.1.1) Quantum execution (KBW) address
  -p [ --port ]  (=4242)     Quantum execution (KBW) port address
  --no-execute               Does not execute the quantum code, measuments 
                             return 0
```

## Installation

> Ket Bitwise Simulator is required for quantum execution. See
> https://gitlab.com/quantum-ket/kbw#installation for installation instructions.

Available installation methods:

* [Snap](#install-using-snap) (recommended)
* [pip](#install-using-pip)
* [Source](#install-from-source)
* AppImage [:arrow_down:](https://gitlab.com/quantum-ket/ket/-/jobs/artifacts/master/download?job=appimage)

### Install using Snap

The ket is available in most Linux distribution through the Snap Store.

>Information on how to enable Snap on your Linux distribution is available on
>https://snapcraft.io/ket.

To install using snap runs:

```console
$ sudo snap install ket --edge
```
> To install Python packages setup a [virtual environment](https://docs.python.org/3/tutorial/venv.html):
> ```console
> $ virtualenv -p /snap/ket/current/usr/bin/python3.8 ~/snap/ket/common
> $ source ~/snap/ket/common/bin/activate
> $ pip install <package>
> ```

### Install using pip

Install requirements:

* C/C++ compiler
* CMake
* Ninja or GNU Make

To install using pip runs:

```console
$ pip install ket-lang
```

You can import Ket source code as a Python module using the `ket.import_ket.import_ket` function.

### Install from source 

> Recommended if you want to be up to date without using Snap.

Install requirements:

* C/C++ compiler
* CMake
* Ninja or GNU Make
* SWIG

To install from source runs:

```console
$ git clone --recurse-submodules https://gitlab.com/quantum-ket/ket.git
$ cd ket
$ make
$ python setup.py install
```

## Run examples

Available examples:

* Quantum phase estimation - [phase_estimation.ket](examples/phase_estimation.ket)
* Quantum teleportation - [teleport.ket](examples/teleport.ket)
* Quantum teleportation using `code_ket` - [teleport.py](examples/teleport.py)
* Random number generation - [random.ket](examples/random.ket)
* Shor's algorithms - [shor15.ket](examples/shor15.ket)
* Shor's algorithms with state dumping - [shor15dump.ket](examples/shor15dump.ket)

With kbw running execute:

```console
$ ket examples/<example>
```

> Replace `<example>` one of the available examples

-----------

This project is part of the Ket Quantum Programming, see the documentation for
more information https://quantum-ket.gitlab.io.
