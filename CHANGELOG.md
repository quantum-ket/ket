# Changelog

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