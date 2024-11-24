<!--
SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>

SPDX-License-Identifier: Apache-2.0
-->

# Changelog

## [0.8.1](https://pypi.org/project/ket-lang/0.8.1)

### Added

- Added Pauli I for Hamiltonian construction.

## [0.8.0](https://pypi.org/project/ket-lang/0.8.0)

### Added

- Added IBM device support.

### Updated

- Updated Libket to 0.5.0.
- Updated KBW to 0.4.0.

## [0.7.1](https://pypi.org/project/ket-lang/0.7.1)

### Added

- Added multi-controlled single-qubit gate decomposition option.
- Added U3 gate, unitary gate constructor, and global phase decorator. 

### Updated

- Updated Libket to 0.4.1.
- Updated KBW to 0.2.1.

## [0.7](https://pypi.org/project/ket-lang/0.7)

### Updated

- Conducted a significant refactoring in the Ket library, eliminating the stack process and introducing functionality for expected value calculations.
- Updated Libket to 0.4.0.
- Updated KBW to 0.2.0.


## [0.6.1](https://pypi.org/project/ket-lang/0.6.1)

### Updated

- Updated Libket to 0.3.1.
- Updated KBW to 0.1.7.

## [0.6](https://pypi.org/project/ket-lang/0.6)

### Added

- Added multi-controlled quantum gate decomposition.

### Updated

- Updated Libket to 0.3.0.
- Updated KBW to 0.1.6.
- Updated Linux CI Rust version to 1.69.

### Removed

- QFT implementation removed from `ket.lib`. You can find implementation examples in [`examples/phase_estimation.py`](examples/phase_estimation.py) and [`examples/shor.py`](examples/shor.py).

## [0.5.3](https://pypi.org/project/ket-lang/0.5.3)

### Updated

- Updated Libket to 0.2.3.
- Updated KBW to 0.1.5.

## [0.5.2](https://pypi.org/project/ket-lang/0.5.2)

### Added

- Added support for `manylinux_2_28_aarch64` wheel.

## [0.5.1](https://pypi.org/project/ket-lang/0.5.1)

### Updated

- Updated KBW: Dump type selection and RNG seed added.
- Update Linux wheel from `manylinux2014` to `manylinux_2_28`.
- Source distribution now packages Libket and KBW source code.

## [0.5.0.1](https://pypi.org/project/ket-lang/0.5.0.1)

### Fixed

- Fixed `lib.qft`.

## [0.5.0](https://pypi.org/project/ket-lang/0.5.0)

### Added

- Added `set_process_features` function.

### Updated

- Libket updated: Dump types Probability and Shots added.

## [0.4.3](https://pypi.org/project/ket-lang/0.4.3)

### Added

- Added macOS support.

## [0.4.2](https://pypi.org/project/ket-lang/0.4.2)

### Fixed

- Fixed calling the inverse of an inverse quantum gate.

## [0.4.1.post1](https://pypi.org/project/ket-lang/0.4.1.post1)

### Added

- Added Windows support.

## [0.4.1](https://pypi.org/project/ket-lang/0.4.1)

### Updated

- Reduced wheel size.
- Fixed `reversed` for `quant`.
- Fixed `quantum_code_last`.
- Fixed triggering empty quantum execution when reading `future.value`.

## [0.4](https://pypi.org/project/ket-lang/0.4)

### Added

- Libket and KBW ported to Rust.
- KBW now features two simulation modes, Dense and Sparse. The Sparse mode is the default, but it may change in the future. Use the environment variable `KBW_MODE` or the functions `ket.kbw.use_sparse()` and `ket.kbw.use_dense` to select the simulation algorithm.
  - The Dense simulation uses the State Vector representation.
  - The Sparse simulation uses the Bitwise representation.
- Libket now supports two intermediary representation schemes, JSON and Binary.
- Added functions `quantum_metrics`, `quantum_metrics_last`, `quantum_code`, `quantum_code_last`, `quantum_exec_time`, and `quantum_exec_timeout`.
- Added gates `flipc` and `phase_on`.

## [0.3.3](https://pypi.org/project/ket-lang/0.3.3)

### Added

- Added method `dump.sphere` that returns a Bloch sphere plot.

## [0.3.2](https://pypi.org/project/ket-lang/0.3.2)

### Fixed

- Fixed a bug that allocated a new qubit when passing an empty list to a quant.
- Fixed an error raised by repeated basis states in `dump.show`.
- Fixed an import error raised by `lib.w`.
- Fixed a Libket bug that limits the measurement to 31 qubits.
- `measure` now splits the result every 63 qubits if measuring more than 64 qubits.

## [0.3.1](https://pypi.org/project/ket-lang/0.3.1)

### Fixed

- Fixed qubit free and future set value.

## [0.3](https://pypi.org/project/ket-lang/0.3)

### Updated

- Refactored Libket and its Python wrapper.
- Libket dropped support for the HTTP API.
- Quantum simulators are now loaded from a shared library.

### Added

- Ket module now includes the KBW simulator.
- KBW is loaded automatically as the default quantum execution target.

## [0.2.1](https://pypi.org/project/ket-lang/0.2.1)

### Updated

- Libket updated to fix segfault when the execution server returns an error.
- Libket updated to unstack process with execution error, allowing further quantum executions.

### Fixed

- Fixed sqrt approximation in `dump.show`.
- Changed `measure` to accept `list[quant]`.

## [0.2](https://pypi.org/project/ket-lang/0.2)

### Added

- Added SSH authentication support for the quantum execution.
- Changed quantum gates to the `quantum_gate` class, allowing composition of quantum gates.

### Updated

- `dump.show` reimplemented in Python to fix error in Jupyter Notebook.

### Fixed

- Fixed `lib.dump_matrix`.

## [0.1.1](https://pypi.org/project/ket-lang/0.1.1)

### Updated

- Changed from Boost.Program_options (Libket, C++) to argparse (Python) to fix segmentation fault with flag `-h`.

## [0.1](https://pypi.org/project/ket-lang/0.1)

- First release. For more information on the design of the Kit see:
  - [Article (ACM Journal on Emerging Technologies in Computing Systems)](https://doi.org/10.1145/3474224)
  - [Master's Thesis (PPGCC-UFSC)](https://repositorio.ufsc.br/handle/123456789/229874)
