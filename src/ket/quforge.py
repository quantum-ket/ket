"""QuForge Interface for Ket

The class :class:~ket.quforge.QuForgeKet enables the use of QuForge, an efficient
qudit simulator, within Ket. This simulator supports expected value calculation
and gradient evaluation, making it a great choice for Quantum Machine Learning.
"""

# SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal
from functools import reduce, partial
from .clib.libket import BatchExecution

try:
    from quforge import quforge as qf
    import torch

    QUFORGE_AVAILABLE = True
except ImportError:
    QUFORGE_AVAILABLE = False


class QuForgeKet(BatchExecution):  # pylint: disable=too-many-instance-attributes
    """QuForge Interface for Ket

    This simulator supports expected value calculation and gradient evaluation,
    making it a great choice for Quantum Machine Learning.

    Args:
         num_qubits: Number of qubits to simulate.
         device: Specifies whether to run the simulation on the CPU or GPU. Defaults to "cpu".
         sparse: If True, uses a sparse representation for improved performance on large systems.
             Defaults to True.
         gradient: Enables gradient evaluation for optimization tasks. Defaults to True.
    """

    def __init__(
        self,
        num_qubits: int,
        device: Literal["cpu", "gpu"] = "cpu",
        sparse: bool = True,
        gradient: bool = True,
    ):
        if not QUFORGE_AVAILABLE:
            raise RuntimeError(
                "QuForge is not available. Please install it with: pip install ket-lang[qml]"
            )

        super().__init__()

        self.initial_state = qf.State(
            "-".join(["0"] * num_qubits), dim=2, device=device
        )

        self.num_qubits = num_qubits
        self.device = device
        self.sparse = sparse
        self.gradient_enabled = gradient

        self.p_z = qf.Z(dim=2, device=device)
        self.p_x = qf.X(dim=2, device=device)
        self.p_y = qf.Y(dim=2, device=device)
        self.p_i = qf.eye(dim=2, device=device, sparse=sparse)

        if sparse:
            self.p_z = self.p_z.M.to_sparse()
            self.p_x = self.p_x.M.to_sparse()
            self.p_y = self.p_y.M.to_sparse()

        self.pauli_map = {
            "X": self.p_x,
            "Y": self.p_y,
            "Z": self.p_z,
            "I": self.p_i,
        }

        self.circuit = None
        self.exp_result = None
        self.gradient = None
        self.parameters = None

    def pauli_x(self, target, control):
        """Apply a Pauli-X gate to the target qubit."""
        assert len(control) <= 1, "Control qubits are not supported"
        if len(control) == 0:
            self.circuit.X(index=[target])
        else:
            self.circuit.CNOT(index=[control[0], target])

    def pauli_y(self, target, control):
        """Apply a Pauli-Y gate to the target qubit."""
        assert len(control) == 0, "Control qubits are not supported"
        self.circuit.Y(index=[target])

    def pauli_z(self, target, control):
        """Apply a Pauli-Z gate to the target qubit."""
        assert len(control) == 0, "Control qubits are not supported"
        self.circuit.Z(index=[target])

    def hadamard(self, target, control):
        """Apply a Hadamard gate to the target qubit."""
        assert len(control) == 0, "Control qubits are not supported"
        self.circuit.H(index=[target])

    def rotation_x(self, target, control, **kwargs):
        """Apply a rotation around the X-axis to the target qubit."""
        assert len(control) == 0, "Control qubits are not supported"
        match kwargs:
            case {"Value": value}:
                ...
            case {"Ref": {"index": index, "multiplier": multiplier, "value": _}}:
                value = self.parameters[index] * multiplier

        self.circuit.RX(index=[target], angle=value)

    def rotation_y(self, target, control, **kwargs):
        """Apply a rotation around the Y-axis to the target qubit."""
        assert len(control) == 0, "Control qubits are not supported"
        match kwargs:
            case {"Value": value}:
                ...
            case {"Ref": {"index": index, "multiplier": multiplier, "value": _}}:
                value = self.parameters[index] * multiplier

        self.circuit.RY(index=[target], angle=value)

    def rotation_z(self, target, control, **kwargs):
        """Apply a rotation around the Z-axis to the target qubit."""
        assert len(control) == 0, "Control qubits are not supported"
        match kwargs:
            case {"Value": value}:
                ...
            case {"Ref": {"index": index, "multiplier": multiplier, "value": _}}:
                value = self.parameters[index] * multiplier

        self.circuit.RZ(index=[target], angle=value)

    def phase(self, target, control, **kwargs):
        """Apply a phase gate to the target qubit."""
        self.rotation_z(target, control, **kwargs)

    def exp_value(self, _, hamiltonian):
        """Compute the expectation value."""
        coef = hamiltonian["coefficients"]
        products = hamiltonian["products"]

        h_final = None

        for pauli, c in zip(products, coef):
            h_partial = ["I" for _ in range(self.num_qubits)]
            for p in pauli:
                h_partial[self.get_qubit_index(p["qubit"])] = p["pauli"][-1]
            h_partial = list(map(lambda x: self.pauli_map[x], h_partial))
            h_partial = c * reduce(partial(qf.kron, sparse=self.sparse), h_partial)

            if h_final is None:
                h_final = h_partial
            else:
                h_final += h_partial

        output_state = self.circuit(self.initial_state)
        expected_value = (output_state.T.conj() @ (h_final @ output_state)).real
        expected_value.backward()
        self.exp_result = float(expected_value[0])

        self.gradient = [p.grad.sum().item() for p in self.parameters]

    def submit_execution(self, logical_circuit, _, parameters):
        self.gradient = [None for _ in range(len(parameters))]
        self.parameters = [
            torch.nn.Parameter(p * torch.ones(self.num_qubits, device=self.device))
            for p in parameters
        ]
        self.process_instructions(logical_circuit)

    def get_result(self):
        return {
            "measurements": [],
            "exp_values": [self.exp_result] if self.exp_result is not None else [],
            "samples": [],
            "dumps": [],
            "gradients": self.gradient if self.gradient_enabled else None,
        }

    def clear(self):
        self.circuit = qf.Circuit(
            dim=2,
            wires=self.num_qubits,
            sparse=self.sparse,
            device=self.device,
        )

        self.exp_result = None
        self.gradient = None
        self.parameters = None

    def config(self):
        """Configures a Process to use QuForge.

        The return value of this function must be passed to a :class:`~ket.base.Process`
        constructor. The result should not be reused, and the QuForgeKet instance must remain alive
        until the end of the process's lifetime.
        """

        self.clear()
        return super().configure(
            num_qubits=self.num_qubits,
            measure="Disable",
            sample="Disable",
            exp_value="Allowed",
            dump="Disable",
            gradient="SupportsGradient" if self.gradient_enabled else "Disable",
            define_qpu=True,
            coupling_graph=None,
            u4_gate_type="CX",
            u2_gate_set="All",
        )
