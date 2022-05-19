# Changelog

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
