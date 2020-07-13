[![Documentation Status](https://readthedocs.org/projects/ket/badge/?version=latest)](https://ket.readthedocs.io/en/latest/?badge=latest)
[![ket](https://snapcraft.io//ket/badge.svg)](https://snapcraft.io/ket)
[![PyPI](https://img.shields.io/pypi/v/ket-lang.svg)](https://pypi.org/project/ket-lang/)


# Ket Quantum Programming Language

Ket is a Python embedded quantum programming language for dynamic interaction
between classical and quantum computers.

## Installation

The Ket Bitwise Simulator is required for quantum execution. It is available
in most Linux distribution through the Snap Store.

Information on how to enable Snap on your Linux distribution is available on
https://snapcraft.io/kbw.

```bash
sudo snap install kbw --edge
```

See https://gitlab.com/quantum-ket/kbw for kbw installation from source.

### Snap

The Ket Quantum Programming Language is also available in the Snap Store.

```bash
sudo snap install ket --edge
```

> **Usage:** `ket <source.ket>`

### PyPI

You can install Ket from using pip as well.

```bash
pip install ket-lang
```
> **Usage:** `python -m ket <source.ket>`

You can import Ket source code as a Python module using the `ket.import_ket.import_ket` function.

### Install from source 

To install from source, follow the commands:

```bash
git clone --recurse-submodules https://gitlab.com/quantum-ket/ket.git
pip install scikit-build
python setup.py install
```

-----------

This project is part of the Ket Quantum Programming, see the documentation for
more information http://ket.readthedocs.io.