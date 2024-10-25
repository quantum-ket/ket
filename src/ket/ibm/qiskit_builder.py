""" Builder for qiskit quantum circuits. """

from __future__ import annotations

# SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2024 Otávio Augusto de Santana Jatobá
# <otavio.jatoba@grad.ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import library, Gate
    from qiskit.quantum_info import SparsePauliOp
except ImportError as exc:
    raise ImportError(
        "QiskitBuilder requires the qiskit module to be used. You can install them"
        "alongside ket by running `pip install ket[ibm]`."
    ) from exc
from typing import Any


class QiskitBuilder:
    """Builder for qiskit quantum circuits from ket-lang instructions."""

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits
        self.circuit = QuantumCircuit(num_qubits, num_qubits)

    def build(self, instructions, meas_map: dict, sample_map: dict) -> dict[str, Any]:
        """Builds the Qiskit quantum circuit from the ket-lang instructions."""

        data = {
            "circuit": self.circuit,
            "observables": [],
        }

        for inst in instructions:
            if "Identity" in inst:
                continue

            if "Gate" in inst:
                gate_type = inst["Gate"]["gate"]
                gate: Gate = self.get_gate(gate_type)

                control = inst["Gate"]["control"]
                if len(control):
                    gate = gate.control(len(control))
                control = [self._get_qubit_index(qubit) for qubit in control]
                data["circuit"].append(
                    gate, control + [self._get_qubit_index(inst["Gate"]["target"])]
                )

            elif "Measure" in inst:
                qubits = [
                    self._get_qubit_index(qubit) for qubit in inst["Measure"]["qubits"]
                ]
                meas_map[inst["Measure"]["index"]] = qubits
                data["circuit"].measure(qubits, qubits)

            elif "ExpValue" in inst:
                hamiltonian: dict = inst["ExpValue"]["hamiltonian"]
                data["observables"].append(self.build_observable(hamiltonian))

            elif "Sample" in inst:
                qubits = [
                    self._get_qubit_index(qubit) for qubit in inst["Sample"]["qubits"]
                ]
                sample_map[inst["Sample"]["index"]] = qubits
                sample_map["shots"] = max(
                    sample_map.get("shots", 2048), inst["Sample"]["shots"]
                )
                data["circuit"].measure(qubits, qubits)

            elif "Dump" in inst:
                raise RuntimeError("Operation not supported")

            else:
                raise RuntimeError("Unknown operation")

        return data

    @staticmethod
    def _get_qubit_index(qubit):
        try:
            return qubit["index"]
        except KeyError:
            return qubit["Main"]["index"]

    def get_gate(self, gate_type) -> Gate:
        """From the type of the gate, returns the corresponding qiskit gate object."""

        if "PauliX" in gate_type:
            return library.XGate()
        if "PauliY" in gate_type:
            return library.YGate()
        if "PauliZ" in gate_type:
            return library.ZGate()
        if "Hadamard" in gate_type:
            return library.HGate()
        if isinstance(gate_type, dict) and any(
            "Rotation" in key or "Phase" in key for key in list(gate_type.keys())
        ):
            angle = list(gate_type.values())[0]
            return self._get_rotation_or_phase_gate(gate_type, angle)

        raise RuntimeError("Unknown gate")

    def _get_rotation_or_phase_gate(self, gate_type, theta) -> Gate:
        """From the type of the gate, returns the corresponding qiskit rotation or
        phase gate object."""

        if "RotationX" in gate_type:
            return library.RXGate(theta)
        if "RotationY" in gate_type:
            return library.RYGate(theta)
        if "RotationZ" in gate_type:
            return library.RZGate(theta)

        return library.U1Gate(theta)  # Phase gate

    def build_observable(self, hamiltonian: dict[str, Any]) -> SparsePauliOp:
        """Builds a Qiskit compliant observable format from the ket-lang hamiltonian
        format.
        ---
        Details:
        -
        The ket hamiltonian format is a dictionary with the following keys:
        - "coefficients": List of coefficients for each product term.
        - "products": List of product terms, where each term is a list of dictionaries
          with the following keys:
            - "qubit": The qubit index.
            - "pauli": The Pauli operator acting on the qubit.

        The qiskit observable is a SparsePauliOp object, made upon a list of Pauli
        strings and a list of coefficients. The Pauli strings are made by concatenating
        the Pauli operators acting on each qubit in the system. The coefficients are
        spread across the Pauli strings, with each corresponding coefficient repeated
        for each Pauli string in the product term.

        Example:
        -
        Given the following ket hamiltonian in a system with 3 qubits:

        hamiltonian = {
            "coefficients": [1, 2],
            "products": [
                [{"qubit": 0, "pauli": "PauliX"}
                {"qubit": 1, "pauli": "PauliX"},
                {"qubit": 1, "pauli": "PauliZ"}],

                [{"qubit": 0, "pauli": "PauliY"}
                {"qubit": 1, "pauli": "PauliY"},
                {"qubit": 2, "pauli": "PauliZ"}],
            ],
        }

        The qiskit observable will be build upon the following components:
        - qiskit_hamiltonian = ["XXI", "IZI, "YYZ"]
        - qiskit_coeffs = [1, 1, 2]

        The qiskit_hamiltonian and qiskit_coeffs are used to create a SparsePauliOp
        object, which is then used to calculate the expectation value.
        """
        hamiltonian_sum: list[list[dict]] = hamiltonian["products"]
        qiskit_hamiltonian = []
        qiskit_coeffs = []

        for coef_index, product in enumerate(hamiltonian_sum):
            qiskit_hmlt_product = []
            i = 0
            while i < len(product):
                qiskit_hamlt_term = ["I"] * self.num_qubits
                while i < len(product):
                    current_qubit: int = self._get_qubit_index(product[i]["qubit"])
                    if qiskit_hamlt_term[current_qubit] != "I":
                        break
                    pauli_type: str = product[i]["pauli"][-1]
                    qiskit_hamlt_term[current_qubit] = pauli_type
                    i += 1
                qiskit_hmlt_product.append("".join(reversed(qiskit_hamlt_term)))

            qiskit_hamiltonian.extend(qiskit_hmlt_product)
            qiskit_coeffs.extend(
                [hamiltonian["coefficients"][coef_index]] * len(qiskit_hmlt_product)
            )

        observable = SparsePauliOp(qiskit_hamiltonian, qiskit_coeffs)
        return observable
