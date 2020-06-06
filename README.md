# Ket quantum programming

Repository for the Ket library for Python and the Ket Interpreter.

## Installation

Dependencies:

* CONAN
* SWIG  

```bash
git clone --recurse-submodules https://gitlab.com/quantum-ket/ket.git
mkdir ket/build
cd ket/build
cmake .. -DPYTHON_PACKAGES=/home/<user>/.local/lib/python3.<minor versions>/site-packages
sudo make install
```

## Run

```bash
python -m keti
```

## Example

```bash
python -m keti examples/teleport.ket
```
