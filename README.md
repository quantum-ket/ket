[![Documentation Status](https://readthedocs.org/projects/ket/badge/?version=latest)](https://ket.readthedocs.io/en/latest/?badge=latest)
[![ket](https://snapcraft.io//ket/badge.svg)](https://snapcraft.io/ket)
[![PyPI](https://img.shields.io/pypi/v/ket-lang.svg)](https://pypi.org/project/ket-lang/)


# Ket quantum programming

Repository for the Ket library for Python and the Ket Interpreter.

## Installation

The Ket Bitwise Simulator is required for quantum execution.
```bash
sudo snap install kbw --edge
```

# pip 

```bash
pip install ket-lang
```

# Snap

```bash
sudo snap install ket --edge
```

# From source

```bash
git clone --recurse-submodules https://gitlab.com/quantum-ket/ket.git
pip install scikit-build
python setup.py install
```

## Execute

# From pip and source installation

```bash
python -m ket example/teleport.ket
```

# From Snap installation

```bash
ket example/teleport.ket
```
