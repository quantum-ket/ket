<!--
SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>

SPDX-License-Identifier: Apache-2.0
-->

# Ket Quantum Programming

[![PyPI](https://img.shields.io/pypi/v/ket-lang.svg)](https://pypi.org/project/ket-lang/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![REUSE status](https://api.reuse.software/badge/gitlab.com/quantum-ket/ket)](https://api.reuse.software/info/gitlab.com/quantum-ket/ket)

Ket is a Python library for quantum programming. You write quantum code as
ordinary Python functions and run it on the built-in simulator, on a GPU, or on
IBM Quantum and Amazon Braket, so it works with NumPy, SciPy and the rest of
the Python ecosystem. Learn more at <https://quantumket.org>.

## Installation ⬇️

Ket needs Python 3.10 or newer, on Linux, Windows or macOS:

```shell
pip install ket-lang
```

On Linux or Windows with a non-x86_64 CPU (such as ARM), install
[Rust](https://www.rust-lang.org/tools/install) first. Optional features:

| Extra              | Adds                                   |
| ------------------ | -------------------------------------- |
| `ket-lang[plot]`   | histograms and circuit drawings        |
| `ket-lang[ibm]`    | IBM Quantum and Qiskit Aer (`ket.ibm`) |
| `ket-lang[amazon]` | Amazon Braket (`ket.amazon`)           |

## Quick start 🚀

```python
from ket import *

p = Process()          # a quantum simulator
a, b = p.alloc(2)      # two qubits in |0⟩

H(a)                   # superposition
with control(a):       # entangle: flip b when a is |1⟩
    X(b)

print(sample(a + b).get())   # {0: ~1024, 3: ~1024}: always 00 or 11

with obs():
    zz = Z(a) * Z(b)
print(exp_value(zz).get())   # 1.0: a and b always agree
```

The main concepts:

- **`Process`** runs the program. It uses the sparse simulator by default; pass
  `simulator="dense"` or `"dense gpu"`, or a device such as
  `ket.ibm.IBMDevice`. In **live** execution (the default) results are ready
  immediately; **batch** execution runs the whole circuit at the end, like a QPU.
- **Gates** are functions: `H(q)`, `CNOT(a, b)`, `RX(θ, q)`. Your own functions
  are gates too, and `adj(f)`, `ctrl(c, f)`, `with control(c):` and
  `with around(f, q):` work on them.
- **Results**: `measure(q).value` (one shot), `sample(q).get()` (counts),
  `exp_value(h).get()` (expectation value), and `dump(q).show()` (state vector).
- **`ket.qulib`** has state preparation, oracles, Hamiltonians and QUBO solvers.

See [`examples/`](examples/) for complete programs, from a Bell pair to QAOA.

## Using an AI coding agent 🤖

Run this in your project to teach AI coding agents (Claude Code, Codex,
Cursor, ...) how to write Ket code:

```shell
ket skills install
```

It adds the Ket skill to `.claude/skills/` and `.agents/skills/`, and a
pointer to it in `AGENTS.md`. Run it again after upgrading Ket to update it.

## Logging 🔎

Ket logs through Python's `logging`. To see what the runtime and simulator do,
such as how gates are decomposed or why an error happened:

```python
import ket
ket.set_log_level("debug")   # or "info", "warning" (default), "error"
```

or set `KET_LOG=debug` in the environment. See [`examples/logs.py`](examples/logs.py).

## Development 🔨

1. Install [Rust](https://www.rust-lang.org/tools/install) and clone the
   repository with its submodules:

   ```shell
   git clone --recursive https://gitlab.com/quantum-ket/ket.git
   cd ket
   ```

2. Install Ket in editable mode, with the development tools:

   ```shell
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements_dev.txt
   pip install -e .
   ```

   The first `import ket` compiles the Rust libraries (Libket and KBW) in
   release mode. On Linux with glibc 2.34 or newer, KBW is built with the GPU
   (`gpu`) and CUDA (`cuda`) simulators; set `KET_BUILD_GPU=0` or
   `KET_BUILD_CUDA=0` to leave them out. To rebuild, delete `libket.so` and
   `libkbw.so` from `src/ket/clib/libs/`.

3. Run the tests:

   ```shell
   pytest
   ```

To build the Rust libraries yourself, for example in debug mode:

```shell
cargo build --manifest-path src/ket/clib/libs/libket/Cargo.toml
cargo build --manifest-path src/ket/clib/libs/kbw/Cargo.toml  # add -F gpu,cuda for GPUs
ln -s libket/target/debug/libket.so src/ket/clib/libs/libket.so
ln -s kbw/target/debug/libkbw.so src/ket/clib/libs/libkbw.so
```

For Rust development, use the same `rustc` version as the current Ket
release.

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
