"""Matrix Product State (MPS) Quantum Simulator Backend.

This module provides the :class:`~ket.mps.MPS` backend, which leverages Tensor Networks
(via ``quimb.tensor``) and JAX's Automatic Differentiation to simulate quantum circuits
with a 1D linear topology. It supports calculating exact expectation values and their gradients,
making it suitable for running parameterized quantum algorithms like VQE efficiently.
"""

# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal
from collections import Counter

from ket.clib.libket.execution import BatchExecution

try:
    import jax
    import jax.numpy as jnp
    import quimb.tensor as qtn
    import numpy as np
except ImportError as e:
    raise ImportError(
        "Ket MPS backend requires optional dependencies."
        "Install them with: pip install 'ket-lang[mps]'"
    ) from e

__all__ = ["MPS"]


def _get_matrix(gate_type: str, val: float = 0.0):
    # pylint: disable=too-many-return-statements
    match gate_type:
        case "Hadamard":
            return jnp.array([[1, 1], [1, -1]], dtype=complex) / jnp.sqrt(2)
        case "PauliX":
            return jnp.array([[0, 1], [1, 0]], dtype=complex)
        case "PauliY":
            return jnp.array([[0, -1j], [1j, 0]], dtype=complex)
        case "PauliZ":
            return jnp.array([[1, 0], [0, -1]], dtype=complex)
        case "RotationX":
            return jnp.array(
                [
                    [jnp.cos(val / 2), -1j * jnp.sin(val / 2)],
                    [-1j * jnp.sin(val / 2), jnp.cos(val / 2)],
                ],
                dtype=complex,
            )
        case "RotationY":
            return jnp.array(
                [
                    [jnp.cos(val / 2), -jnp.sin(val / 2)],
                    [jnp.sin(val / 2), jnp.cos(val / 2)],
                ],
                dtype=complex,
            )
        case "RotationZ":
            return jnp.array(
                [[jnp.exp(-1j * val / 2), 0], [0, jnp.exp(1j * val / 2)]], dtype=complex
            )
        case "Phase":
            return jnp.array([[1, 0], [0, jnp.exp(1j * val)]], dtype=complex)
    raise ValueError(f"Unsupported gate type: {gate_type}")


def _controlled_matrix(matrix, num_controls: int):
    size = 2 ** (num_controls + 1)
    mat = jnp.eye(size, dtype=complex)
    if matrix.ndim == 2:
        mat = mat.at[-2:, -2:].set(matrix)
    elif matrix.ndim == 4:
        mat = mat.at[-4:, -4:].set(matrix.reshape((4, 4)))
    return mat.reshape((2,) * 2 * (num_controls + 1))


def _build_psi(params_array, gates_tuple, num_qubits, max_bond, force_contract=False):
    psi = qtn.MPS_computational_state("0" * num_qubits)
    if not force_contract:
        psi.apply_to_arrays(jnp.array)

    for g_type, v_idx, v_mult, v_val, qbs in gates_tuple:
        if force_contract:
            val = float(params_array[v_idx] * v_mult) if v_idx >= 0 else float(v_val)
        else:
            val = jnp.where(v_idx >= 0, params_array[v_idx] * v_mult, v_val)

        mat = _get_matrix(g_type, val)
        if len(qbs) > 1:
            mat = _controlled_matrix(mat, len(qbs) - 1)

        if force_contract:
            mat = np.array(mat)

        kwargs = {"inplace": False}
        if len(qbs) > 1:
            if max_bond is not None:
                kwargs["contract"] = "split"
                kwargs["max_bond"] = max_bond
            elif force_contract:
                kwargs["contract"] = "swap+split"
        elif force_contract or max_bond is not None:
            kwargs["contract"] = True

        psi = psi.gate(mat, qbs, **kwargs)
    return psi


def _build_and_evaluate(
    params_array,
    gates_tuple,
    hamiltonian_tuple,
    num_qubits,
    max_bond,
):
    # pylint: disable=too-many-locals
    psi = _build_psi(
        params_array,
        gates_tuple,
        num_qubits,
        max_bond,
        force_contract=False,
    )

    results = []
    for h_coeffs, h_strings in hamiltonian_tuple:
        total_exp = 0.0
        for coeff, p_string in zip(h_coeffs, h_strings):
            psi_h = psi
            for p_type, p_qubit in p_string:
                p_mat = _get_matrix(p_type)
                psi_h = psi_h.gate(p_mat, (p_qubit,), inplace=False)

            exp_val = (psi.H @ psi_h).real
            total_exp += coeff * exp_val
        results.append(total_exp)

    if len(results) > 0:
        return results[0]
    return 0.0


_jitted_build_and_evaluate = jax.jit(_build_and_evaluate, static_argnums=(1, 2, 3, 4))


class MPS(BatchExecution):
    """Matrix Product State (MPS) backend simulator.

    The ``MPS`` backend is specialized for executing quantum circuits with 1D linear topologies.
    It uses tensor network contractions via the ``quimb`` library, augmented by JAX to support
    exact parameter gradient evaluations for parameterized circuits.

    Args:
        num_qubits: Total number of qubits to simulate.
        backend: Target hardware for execution, either ``"cpu"`` or ``"gpu"``.
            Defaults to ``"cpu"``.
        max_bond: Maximum bond dimension for SVD truncations during the simulation.
            If specified, it restricts entanglement and keeps the MPS strictly 1D, enabling
            the simulation of large numbers of qubits. However, this dynamically changes tensor
            shapes and disables JAX's ``jit`` compilation, which may result in slower evaluations.
            If ``None``, the network is built in 2D and contracted fully at the end,
            which is JIT-compatible but may scale poorly with depth. Defaults to ``None``.
    """

    def __init__(
        self,
        num_qubits: int,
        backend: Literal["cpu", "gpu"] = "cpu",
        max_bond: int = None,
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.max_bond = max_bond
        if backend not in ["cpu", "gpu"]:
            raise ValueError("backend must be 'cpu' or 'gpu'")
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_platform_name", backend)

        self.last_grad = None

    def connect(self):
        """Configure the connection to the MPS simulator."""
        if self.max_bond is not None:
            coupling_graph = [(i, i + 1) for i in range(self.num_qubits - 1)]
        else:
            coupling_graph = None

        return self.configure(
            self.num_qubits,
            gradient="Native",
            coupling_graph=coupling_graph,
            decompose=True,
        )

    def sample(self, gates, qubits_to_sample, shots):
        # pylint: disable=too-many-locals
        gates_tuple_list = []
        for g in gates:
            gate_info = g["gate"]
            target = g.get("target", 0)
            controls = g.get("control", [])

            if isinstance(gate_info, str):
                gate_type = gate_info
                v_val = 0.0
            else:
                gate_type = list(gate_info.keys())[0]
                val_info = gate_info[gate_type]
                v_val = val_info.get("Value", 0.0)

            qbs = tuple(controls) + (target,)

            gates_tuple_list.append((gate_type, -1, 0.0, v_val, qbs))

        gates_tuple = tuple(gates_tuple_list)

        psi = _build_psi(
            jnp.array([0.0]),
            gates_tuple,
            self.num_qubits,
            self.max_bond,
            force_contract=True,
        )

        samples = list(psi.sample(shots))

        counts = Counter()
        for sample_res in samples:
            # sample_res is like ([0, 1, 0, 1], prob)
            sample_state = (
                sample_res[0]
                if isinstance(sample_res, tuple)
                and len(sample_res) == 2
                and isinstance(sample_res[0], list)
                else sample_res
            )

            bits = list(sample_state)

            val = 0
            for i, q in enumerate(qubits_to_sample):
                if bits[q]:
                    val |= 1 << i
            counts[val] += 1

        return [[int(k)] for k in counts.keys()], [int(v) for v in counts.values()]

    def gradient(self, gates, hamiltonian):
        # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        param_values = {}
        gates_tuple_list = []

        for g in gates:
            gate_info = g["gate"]
            target = g.get("target", 0)
            controls = g.get("control", [])

            if isinstance(gate_info, str):
                gate_type = gate_info
                val_idx = -1
                val_mult = 0.0
                v_val = 0.0
            else:
                gate_type = list(gate_info.keys())[0]
                val_info = gate_info[gate_type]
                if isinstance(val_info, dict) and "Ref" in val_info:
                    val_idx = val_info["Ref"]["index"]
                    val_mult = val_info["Ref"]["multiplier"]
                    param_values[val_idx] = val_info["Ref"]["value"]
                    v_val = 0.0
                else:
                    val_idx = -1
                    val_mult = 0.0
                    v_val = val_info.get("Value", 0.0)

            qbs = tuple(controls) + (target,)

            gates_tuple_list.append((gate_type, val_idx, val_mult, v_val, qbs))

        gates_tuple = tuple(gates_tuple_list)

        h_tuple_list = []
        for h in [hamiltonian]:
            coeffs = tuple(h["coefficients"])
            strings = tuple(
                tuple((p["pauli"], p["qubit"]) for p in s) for s in h["pauli_strings"]
            )
            h_tuple_list.append((coeffs, strings))
        h_tuple = tuple(h_tuple_list)

        if param_values:
            max_idx = max(param_values.keys())
            params_arr = jnp.zeros(max_idx + 1)
            for k, v in param_values.items():
                params_arr = params_arr.at[k].set(v)

            eval_fn = (
                _build_and_evaluate
                if self.max_bond is not None
                else _jitted_build_and_evaluate
            )
            val_grad_fn = jax.value_and_grad(eval_fn)
            exp_val, grad = val_grad_fn(
                params_arr, gates_tuple, h_tuple, self.num_qubits, self.max_bond
            )

            self.last_grad = grad.tolist()
            return float(exp_val), grad.tolist()

        eval_fn = (
            _build_and_evaluate
            if self.max_bond is not None
            else _jitted_build_and_evaluate
        )
        exp_val = eval_fn(
            jnp.array([0.0]), gates_tuple, h_tuple, self.num_qubits, self.max_bond
        )
        return float(exp_val), []
