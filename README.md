[![Documentation Status](https://readthedocs.org/projects/ket/badge/?version=latest)](https://ket.readthedocs.io/en/latest/?badge=latest)
[![ket](https://snapcraft.io//ket/badge.svg)](https://snapcraft.io/ket)
[![PyPI](https://img.shields.io/pypi/v/ket-lang.svg)](https://pypi.org/project/ket-lang/)


# Ket Quantum Programming Language

Ket is a Python-embedded classical-quantum programming language for dynamic
interaction between classical and quantum computers.

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
def bell(aux0, aux1):
    q = quant(2)
    if aux0 == 1:
        x(q[0])
    if aux1 == 1:
        x(q[1])
    h(q[0])
    ctrl(q[0], x, q[1])
    return q

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

a = quant(1)    # a = |0>
h(a)            # a = |+> 
z(a)            # a = |->
y = teleport(a) # y <- a
h(y)            # y = |1>
print('Expected measure 1, result =', measure(y).get())
```

> Expected measure 1, result = 1

### Shor's Algorithm Factoring 15:

```python
from math import pi, gcd
from ket import plugins

def qft(q): # Quantum Fourier Transformation
    lambd = lambda k : pi*k/2
    for i in range(len(q)):
        for j in range(i):
            ctrl(q[i], u1, lambd(i-j), q[j])
        h(q[i])

def period():
    reg1 = quant(4)
    h(reg1)
    reg2 = plugins.pown(7, reg1, 15)
    qft(reg1)
    return measure(reg1.inverted()).get()

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

> measurements = [12, 0, 12, 8, 0]\
> r = 4.0\
> 15 = 5 x 3

## Usage 

```shell
Ket program options:
  -h [ --help ]         Show this information
  -s [ --seed ]         Pseudo random number generator seed
  -o [ --out ]          kqasm output file
  -b [ --kbw ]          Path to the Ket Bitwise simulator
  -p [ --plugin ]       Ket Bitwise plugin directory path
  --no-execute          Do not execute the quantum code, measuments will return
                        0
```

## Installation

The Ket Bitwise Simulator is required for quantum execution. It is available
in most Linux distribution through the Snap Store.

> Information on how to enable Snap on your Linux distribution is available on
https://snapcraft.io/kbw.

```shell
sudo snap install kbw --edge
```

See https://gitlab.com/quantum-ket/kbw for kbw installation from source.

### Snap

The Ket Quantum Programming Language is available in the Snap Store.

```shell
sudo snap install ket --edge
```

> **Usage:** `ket <source.ket>`

### PyPI

You can install Ket from using pip as well.

```shell
pip install ket-lang
```
> **Usage:** `python -m ket <source.ket>`

You can import Ket source code as a Python module using the `ket.import_ket.import_ket` function.

### Install from source 

To install from source, follow the commands:

```shell
git clone --recurse-submodules https://gitlab.com/quantum-ket/ket.git
cd ket
make
python setup.py install
```

-----------

This project is part of the Ket Quantum Programming, see the documentation for
more information https://quantum-ket.gitlab.io.
