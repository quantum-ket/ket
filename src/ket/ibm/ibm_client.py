""" Client for connecting the Qiskit ket interface with the Qiskit runtime backend."""

from __future__ import annotations

# SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2024 Otávio Augusto de Santana Jatobá
# <otavio.jatoba@grad.ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0

try:
    from qiskit import QuantumCircuit
    from qiskit.providers import Backend
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, Session
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
except ImportError as exc:
    raise ImportError(
        "QiskitClient requires the qiskit module to be used. You can install them"
        "alongside ket by running `pip install ket[ibm]`."
    ) from exc
from typing import Any
from .qiskit_builder import QiskitBuilder


class IBMClient:
    """Client for connecting the Qiskit ket interface with the Qiskit runtime backend.

    This class is responsible for processing the instructions received from the ket
    interface, sending them to the Qiskit runtime backend, and formatting the results
    to be sent back to the ket interface.

    Attributes:
        backend: The backend to be used for the quantum circuit.
        service: The Qiskit runtime service to be used for the quantum circuit.
        qiskit_builder: The QiskitBuilder object to build the quantum circuit.

    Methods:
        process_instructions: Processes the instructions received from the ket interface.
    """

    def __init__(
        self, num_qubits: int, backend: Backend, service: QiskitRuntimeService | None
    ) -> None:
        self.backend = backend
        self.service = service
        self.qiskit_builder = QiskitBuilder(num_qubits)

    def process_instructions(
        self, instructions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Processes the instructions received from the ket interface and returns the
        libket compliant formatted results."""

        raw_results = {"samples": [], "exp_values": []}
        measurement_map = {}
        sample_map = {}
        builder_data = self.qiskit_builder.build(
            instructions, measurement_map, sample_map
        )
        pm = generate_preset_pass_manager(
            target=self.backend.target, optimization_level=1
        )
        isa_circuit = pm.run(builder_data["circuit"])

        with Session(service=self.service, backend=self.backend) as session:
            if sample_map or measurement_map:
                result_data = (
                    Sampler(session=session).run([isa_circuit]).result()[0].data
                )
                result = (
                    result_data.c.get_counts()
                    if hasattr(result_data, "c")
                    else result_data.cr.get_counts()
                )
                raw_results["samples"] = result

            # BUG: The following code is not working as expected. The EstimatorV2
            # class is not always returning the expected results.
            for observable in builder_data["observables"]:
                isa_observable = observable.apply_layout(isa_circuit.layout)
                result = (
                    Estimator(session=session)
                    .run([(isa_circuit, isa_observable)])
                    .result()[0]
                    .data.evs
                ).tolist()

                # Calling tolist() on an ndarray with a single element returns a
                # scalar, so we need to check if the result is a list or a scalar.
                if isinstance(result, list):
                    raw_results["exp_values"].extend(result)
                else:
                    raw_results["exp_values"].append(result)

        return self.format_result(raw_results, measurement_map, sample_map)

    def format_result(
        self, raw_results: dict[str, Any], meas_map: dict, sample_map: dict
    ) -> dict[str, Any]:
        """Formats the raw IBM results into a libket compliant dictionary format."""

        result_dict = {
            "measurements": [0] * len(meas_map),
            "exp_values": [],
            "samples": [0] * len(sample_map),
            "dumps": [],
            "execution_time": None,
        }

        if meas_map:
            meas_bin = max(raw_results["samples"], key=raw_results["samples"].get)
            reversed_meas_bin = meas_bin[::-1]

            # meas_index is the index of the measurement in the measurement map, as
            # there can be multiple measurements in a single instruction.
            # qubits is the list of qubits that were measured.
            for meas_index, qubits in meas_map.items():
                result_dict["measurements"][meas_index] = int(
                    "".join([reversed_meas_bin[qubit] for qubit in qubits]), 2
                )

        if sample_map:
            for sample_index, qubits in sample_map.items():
                result_dict["samples"][sample_index] = (
                    [
                        int("".join([measurement[::-1][i] for i in qubits]), 2)
                        for measurement in raw_results["samples"].keys()
                    ],
                    list(raw_results["samples"].values()),
                )

        result_dict["exp_values"].extend(raw_results["exp_values"])
        return result_dict

    @property
    def circuit(self) -> QuantumCircuit:
        """IBM quantum circuit object used in the device."""
        return self.qiskit_builder.circuit
