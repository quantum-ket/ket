---
name: ket
description: Write, run and debug quantum programs with Ket (the `ket-lang` Python package). Use when code imports `ket`, or when asked to build quantum circuits, algorithms (Grover, QFT, phase estimation, QAOA, VQE), observables or expectation values in this project.
---

# Ket quantum programming

Ket is a Python library: gates are function calls on qubits, and the program
runs on the KBW simulator (or a QPU) as it is built. Prefer `from ket import *`.

## Core pattern

```python
from ket import *

p = Process()                  # simulator; see "Process" below
q = p.alloc(3)                 # Quant: q[0], q[1:], q[0] + q[2], len(q)

H(q[0])                        # gates return their qubits: CNOT(H(a), b)
with control(q[0]):            # controlled block
    X(q[1])

print(sample(q).get())         # {state: count}, 2048 shots
print(measure(q).value)        # one shot, collapses the state (live only)
print(dump(q).show())          # full state vector (simulator only)
```

- `q[0]` is the **most significant** bit: after `X(q[2])` on 3 qubits,
  `measure(q).value == 1`.
- Results are integers; for bit strings use `.bitstring`:
  `measure(q).bitstring` gives `'001'`, `sample(q).bitstring` gives
  `{'001': count, ...}`.

## Process

```python
Process()                                   # sparse, live, up to 32 qubits
Process(simulator="dense", num_qubits=20)   # dense: memory 2**n, faster for deep circuits
Process(execution="batch")                  # QPU-like: runs once at the end
Process(gradient=True)                      # p.param(...) gradients; always batch
```

- Simulators: `"sparse"` (few nonzero amplitudes), `"dense"` (general),
  `"dense gpu"`, `"sparse gpu"`, `"dense cuda"`, `"sparse cuda"`.
- Set `num_qubits` for dense simulators (default 12); `alloc` fails when they
  run out.
- **Live** (default): every result is available immediately; classical code can
  branch on `measure(...).value`.
- **Batch**: `measure` raises; use `sample`/`exp_value`. After the first
  result is read (`.get()`), no more gates can be added: build a new `Process`.
- Qubits from different processes cannot be mixed. Create a new `Process` per
  circuit run (for example, per optimizer step).
- QPUs: `Process(IBMDevice(...))` from `ket.ibm` (`pip install ket-lang[ibm]`),
  `Process(AmazonBraket(...))` from `ket.amazon` (`ket-lang[amazon]`). Both
  are batch only.

## Gates

- One qubit: `I X Y Z H S SD T TD SX`, and `RX RY RZ P` (angle first),
  `U3(theta, phi, lambda_, q)`. On a multi-qubit `Quant`, they apply to each
  qubit: `H(q)`.
- Two qubits: `CNOT(c, t)`, `SWAP(a, b)`, `RXX RYY RZZ RBS(theta, a, b)`.
  `CZ(*q)` is a multi-controlled Z.
- Parametric gates curry: `RX(theta)` is a gate, so `ctrl(c, RX(theta))(q)`.
- `QFT(q)`, `evolve(hamiltonian)`.

## Composing gates

```python
with control(c):             X(t)       # controlled on |1...1⟩
with control(c, state=0b01): X(t)       # controlled on another basis state
ctrl(c, SWAP)(a, b)                     # controlled gate as a function
adj(QFT)(q)                             # inverse of any gate function
with around(cat(H, X), q):  CZ(*q)      # U · body · U†  (Grover diffusion)
cat(H, X)(q); kron(H, X)(a, b)          # sequence; parallel
```

A "gate" is any Python function that takes qubits and applies gates. `adj`,
`ctrl`, `control` and `around` work on such functions, so write subroutines as
plain functions. Do not measure inside them.

Put classical arguments first and qubits last, like `RX(theta, q)`, so
`partial` gives a gate on qubits only:

```python
from functools import partial

def layer(gamma, beta, q):         # classical arguments, then qubits
    RZZ(gamma, q[0], q[1])
    RX(beta, q)

adj(partial(layer, 0.1, 0.2))(q)
```

## Observables and expectation values

```python
with obs():                        # inside obs(), X/Y/Z are Pauli operators
    h = 0.5 * Z(q[0]) * Z(q[1]) - X(q[0])
print(exp_value(h).get())          # float; exact on simulators
```

Outside `obs()`, `X(q)` applies a gate. Ready-made Hamiltonians are in
`ket.qulib.ham` (`maxcut(edges, q)`, `x_mixer(q)`, ...).

`obs(n)` gives `n` placeholder qubits, so a Hamiltonian can be built once,
without a process, and bound to the process qubits later:

```python
with obs(2) as o:                  # placeholders o[0], o[1]
    zz = Z(o[0]) * Z(o[1])

p = Process()
q = p.alloc(2)
exp_value(zz, q).get()             # o[i] -> q[i]; same as exp_value(zz(q))
evolve(zz, q)                      # e^{-i zz}
```

## Gradients

```python
p = Process(gradient=True)
theta = p.param(0.1)               # or p.param(0.1, 0.2, ...) for a list
q = p.alloc()
RX(theta, q)
with obs():
    h = Z(q)
ev = exp_value(h); ev.get()
print(theta.grad)                  # d⟨h⟩/dθ
```

Gradients need batch execution: `gradient=True` runs in batch, and
`execution="live"` with it raises `ValueError`. So `measure` is not available,
the gradients are ready after the first `.get()`, and each new set of
parameter values needs a new `Process`.

## Library (`ket.qulib`)

- `qulib.prepare`: `ghz`, `w`, `bell`, `dicke`, `state`, `pauli_state`.
- `qulib.oracle`: `phase_oracle(state, q)`, `xor_oracle`.
- `qulib.gates`: `unitary`, `diagonal`, `ucz`.
- `qulib.ham`, `qulib.qubo`: Hamiltonians and QUBO models;
  `qulib.exact_solver`, `simulated_annealing`, `parallel_tempering`.
- `qulib.gate_count(gate, num_qubits)`: gates and CNOT depth after decomposition.
- `qulib.draw(gate, num_qubits)` (needs `ket-lang[plot]`), `qulib.dump_matrix`.
- `qasm(q, source)`: apply an OpenQASM 2.0 circuit.

## Debugging

- `ket.set_log_level("info")` (or `KET_LOG=debug`) shows what the runtime and
  simulator do; errors are logged with their cause (which qubit, which line).
- `LibketError` messages are short; read the `ERROR` log record for details.
- Common errors: "cannot apply gates after a sample..." (batch process already
  ran: create a new `Process`); "no qubit left to allocate" (raise
  `num_qubits`); "a qubit cannot be both a control and the target" (a control
  qubit is also used inside the `control` block).

## Example: Grover

```python
from math import pi, sqrt
from ket import *

def grover(n, oracle):
    p = Process(simulator="dense", num_qubits=n)
    q = H(p.alloc(n))
    for _ in range(int(pi / 4 * sqrt(2**n))):
        oracle(q)
        with around(cat(H, X), q):
            CZ(*q)
    return measure(q).value

print(grover(8, qulib.oracle.phase_oracle(13)))  # 13
```

More: `help(ket)`, docstrings of each function, and https://quantumket.org.
