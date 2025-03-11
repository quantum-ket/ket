# SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

import math
from .clib.libket import BatchExecution
from functools import reduce, partial
from typing import Literal
from quforge import quforge as qf
import torch


class QuForgeKet(BatchExecution):
    def __init__(
        self,
        num_qubits: int,
        device: Literal["cpu", "gpu"] = "cpu",
        sparse: bool = True,
        gradient: bool = True,
    ):
        super().__init__()
        self.circuit = qf.Circuit(
            dim=2,
            wires=num_qubits,
            sparse=sparse,
            device=device,
        )

        self.initial_state = qf.State(
            "-".join(["0"] * num_qubits), dim=2, device=device
        )

        self.num_qubits = num_qubits
        self.device = device
        self.sparse = sparse
        self.gradient_enabled = gradient

        self.p_z = qf.Z(dim=2, device=device).M
        if sparse:
            self.p_z = self.p_z.to_sparse()
        self.p_x = qf.X(dim=2, device=device).M
        if sparse:
            self.p_x = self.p_x.to_sparse()
        self.p_y = (self.p_z @ self.p_x) / 1j
        self.p_i = qf.eye(dim=2, device=device, sparse=sparse)

        self.pauli_map = {
            "X": self.p_x,
            "Y": self.p_y,
            "Z": self.p_z,
            "I": self.p_i,
        }

        self.exp_result = None
        self.gradient_index = {}
        self.gradient = None
        self.parameters = None

    def pauli_x(self, target, control):
        """Apply a Pauli-X gate to the target qubit."""
        assert len(control) <= 1, "Control qubits are not supported"
        if control is None:
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
                self.gradient_index[len(self.circuit.circuit)] = (
                    index,
                    multiplier,
                )

        self.circuit.RX(index=[target], angle=value)

    def rotation_y(self, target, control, **kwargs):
        """Apply a rotation around the Y-axis to the target qubit."""
        assert len(control) == 0, "Control qubits are not supported"
        match kwargs:
            case {"Value": value}:
                ...
            case {"Ref": {"index": index, "multiplier": multiplier, "value": value}}:
                value = self.parameters[index] * multiplier
                self.gradient_index[len(self.circuit.circuit)] = (
                    index,
                    multiplier,
                )

        self.circuit.RY(index=[target], angle=value)

    def rotation_z(self, target, control, **kwargs):
        """Apply a rotation around the Z-axis to the target qubit."""
        assert len(control) == 0, "Control qubits are not supported"
        match kwargs:
            case {"Value": value}:
                ...
            case {"Ref": {"index": index, "multiplier": multiplier, "value": value}}:
                value = self.parameters[index] * multiplier
                self.gradient_index[len(self.circuit.circuit)] = (
                    index,
                    multiplier,
                )
        self.circuit.RZ(index=[target], angle=value)

    def phase(self, target, control, **kwargs):
        """Apply a phase gate to the target qubit."""
        assert len(control) == 0, "Control qubits are not supported"
        match kwargs:
            case {"Value": value}:
                ...
            case {"Ref": {"index": index, "multiplier": multiplier, "value": value}}:
                value = self.parameters[index] * multiplier
                self.gradient_index[len(self.circuit.circuit)] = (
                    index,
                    multiplier,
                )

        p_gate = torch.tensor([[1, 0], [0, torch.exp(1j * value)]])
        self.circuit.Custom(p_gate, index=[target])

    def exp_value(self, hamiltonian):
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
                h_final = h_final + h_partial

        output_state = self.circuit(self.initial_state)
        expected_value = (output_state.T.conj() @ (h_final @ output_state)).real
        expected_value.backward()
        self.exp_result = float(expected_value[0])

        for gate_index, (index, multiplier) in self.gradient_index.items():
            grad = self.circuit.circuit[gate_index].angle.grad
            grad = math.sqrt(grad[0] ** 2 + grad[1] ** 2) * math.sqrt(multiplier)

            self.gradient[index] += float(grad)

    def submit_execution(self, logical_circuit, _, parameters):
        self.gradient = [0 for _ in range(len(parameters))]
        self.parameters = parameters
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

    def configure(self):
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


def quforge(
    num_qubits: int,
    device: Literal["cpu", "gpu"] = "cpu",
    sparse: bool = True,
    gradient: bool = True,
):
    qf = QuForgeKet(num_qubits, device, sparse, gradient)
    return qf.configure()
