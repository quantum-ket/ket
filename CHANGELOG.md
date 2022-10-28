# Changelog

## 0.5.0

- Added `set_process_features` function.
- Libket updated: Dump types Probability and Shots added;

## 0.4.3

- Added macOS support.

## 0.4.2

- Fixed calling the inverse of a inverse quantum gate.

## 0.4.1.post1

- Added Windows support.

## 0.4.1

- Reduced wheel size.
- Fixed `reversed` for `quant`.
- Fixed `quantum_code_last`.
- Fixed triggering empty quantum execution when reading `future.value`.

## 0.4

- Libket and KBW ported to Rust.
- KBW now features two simulation modes, Dense and Sparse. The Sparse mode is the default, but it may change in the future. Use the environment variable `KBW_MODE` or the functions `ket.kbw.use_sparse()` and `ket.kbw.use_dense` to select the simulation algorithm.
  - The Dense simulation uses the State Vector representation.
  - The Sparse simulation uses the Bitwise representation.
- Libket now supports two intermediary representation schemes, JSON and Binary.
- Added functions `quantum_metrics`, `quantum_metrics_last`, `quantum_code`, `quantum_code_last`, `quantum_exec_time`, and `quantum_exec_timeout`.
- Added gates `flipc` and `phase_on`.

## 0.3.3

- Added method `dump.sphere` that returns a Bloch sphere plot.

## 0.3.2

- Fixed bug that allocate a new qubit when passing an empty list to a quant.
- Fixed error raised by repeated basis states in `dump.show`.
- Fixed import error raised by `lib.w`.
- Fixed Libket bug that limits the measurement to 31 qubits.
- `measure` now splits the result every 63 qubits if measuring more than 64 qubits.

## 0.3.1

- Fixed qubit free and future set value.

## 0.3

- Libket and its Python wrapper wore refactored. Libket dropped support for the HTTP API.  Quantum simulators are now loaded from a shared library.
- Ket module now includes the KBW simulator. KBW is loaded automatically as the default quantum execution target.

## 0.2.1

- Libket updated to fix segfault when the execution server returns an error.
- Libket updated to unstack process with execution error, allowing further quantum executions.
- Fixed sqrt approximation in `dump.show`.
- Changed `measure` to accept `list[quant]`.

## 0.2

- Added SSH authentication support for the quantum execution.
- Changed quantum gates to the `quantum_gate` class, allowing composition of quantum gates.
- `dump.show` reimplemented in Python to fix error in Jupyter Notebook.
- Fixed lib.dump_matrix.

## 0.1.1

- Changed from Boost.Program_options (Libket, C++) to argparse (Python) to fix segmentation fault with flag `-h`.
