"""Native gate set decompositions for Ket."""

# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
# SPDX-FileCopyrightText: 2026 Ruan Luiz Molgero Lopes <ruan.molgero@grad.ufsc.br>
#
# SPDX-License-Identifier: Apache-2.0


import cmath
import math
from .clib.libket.execution import NativeGateSet


def _extract_u3(matrix):
    """
    Extracts U3 angles (theta, phi, lambda) from a 2x2 unitary matrix.
    Ignores global phase.
    """
    u00, u01 = matrix[0]
    u10, u11 = matrix[1]

    if abs(u00) > 1e-10:
        phase_g = cmath.phase(u00)
        u00 *= cmath.exp(-1j * phase_g)
        u01 *= cmath.exp(-1j * phase_g)
        u10 *= cmath.exp(-1j * phase_g)
        u11 *= cmath.exp(-1j * phase_g)

        val = max(-1.0, min(1.0, u00.real))
        theta = 2 * math.acos(val)

        phi = cmath.phase(u10) if abs(u10) > 1e-10 else 0.0
        lam = cmath.phase(-u01) if abs(u01) > 1e-10 else cmath.phase(u11) - phi
    else:
        theta = math.pi
        phase_g = cmath.phase(u10)
        u10 *= cmath.exp(-1j * phase_g)
        u01 *= cmath.exp(-1j * phase_g)
        phi = 0.0
        lam = cmath.phase(-u01) if abs(u01) > 1e-10 else 0.0

    return theta, phi, lam


class RzSXCZ(NativeGateSet):
    """Gate set using CZ for CNOT decomposition."""

    def translate(self, matrix: list[list[complex]], target: int) -> list:
        theta, phi, lam = _extract_u3(matrix)
        return [
            ["rz", [lam], [target]],
            ["sxdg", [], [target]],
            ["rz", [-theta], [target]],
            ["sx", [], [target]],
            ["rz", [phi], [target]],
        ]

    def cnot(self, control: int, target: int) -> list:
        return [
            ["rz", [math.pi / 2], [target]],
            ["sx", [], [target]],
            ["rz", [math.pi / 2], [target]],
            ["cz", [], [control, target]],
            ["rz", [math.pi / 2], [target]],
            ["sx", [], [target]],
            ["rz", [math.pi / 2], [target]],
        ]


class RzSXXRzz(NativeGateSet):
    """Gate set using Rzz for CNOT decomposition."""

    def translate(self, matrix: list[list[complex]], target: int) -> list:
        theta, phi, lam = _extract_u3(matrix)
        return [
            ["rz", [lam], [target]],
            ["sxdg", [], [target]],
            ["rz", [-theta], [target]],
            ["sx", [], [target]],
            ["rz", [phi], [target]],
        ]

    def cnot(self, control: int, target: int) -> list:
        return [
            ["rz", [math.pi / 2], [target]],
            ["sx", [], [target]],
            ["rz", [math.pi / 2], [target]],
            ["rzz", [math.pi / 2], [control, target]],
            ["rz", [math.pi / 2], [target]],
            ["sx", [], [target]],
            ["rz", [math.pi / 2], [target]],
            ["sx", [], [target]],
            ["x", [], [target]],
            ["rz", [-math.pi / 2], [control]],
        ]


class RzSXECR(NativeGateSet):
    """Gate set using ECR for CNOT decomposition."""

    def translate(self, matrix: list[list[complex]], target: int) -> list:
        theta, phi, lam = _extract_u3(matrix)
        return [
            ["rz", [lam], [target]],
            ["sxdg", [], [target]],
            ["rz", [-theta], [target]],
            ["sx", [], [target]],
            ["rz", [phi], [target]],
        ]

    def cnot(self, control: int, target: int) -> list:
        return [
            ["rz", [-math.pi / 2], [control]],
            ["rz", [-math.pi], [target]],
            ["sx", [], [target]],
            ["rz", [-math.pi], [target]],
            ["ecr", [], [control, target]],
            ["sx", [], [control]],
            ["sx", [], [control]],
        ]


class RzRxISWAP(NativeGateSet):
    """Gate set using iSWAP for CNOT decomposition."""

    def translate(self, matrix: list[list[complex]], target: int) -> list:
        theta, phi, lam = _extract_u3(matrix)
        return [
            ["rz", [lam], [target]],
            ["rx", [math.pi / 2], [target]],
            ["rz", [theta], [target]],
            ["rx", [-math.pi / 2], [target]],
            ["rz", [phi], [target]],
        ]

    def cnot(self, control: int, target: int) -> list:
        return [
            ["rz", [-math.pi / 2], [control]],
            ["rz", [math.pi / 2], [target]],
            ["iswap", [], [control, target]],
            ["rx", [math.pi / 2], [control]],
            ["iswap", [], [control, target]],
            ["rx", [-math.pi / 2], [target]],
            ["rz", [math.pi / 2], [target]],
            ["rx", [math.pi / 2], [target]],
            ["rz", [math.pi / 2], [target]],
        ]


class RzGPiGPi2MS(NativeGateSet):
    """Gate set using MS and GPi for CNOT decomposition."""

    def translate(self, matrix: list[list[complex]], target: int) -> list:
        theta, phi, lam = _extract_u3(matrix)
        return [
            ["rz", [lam], [target]],
            ["gpi2", [0.0], [target]],
            ["rz", [theta], [target]],
            ["gpi2", [math.pi], [target]],
            ["rz", [phi], [target]],
        ]

    def cnot(self, control: int, target: int) -> list:
        return [
            ["h", [], [target]],
            ["gpi", [0.0], [control]],
            ["h", [], [control]],
            ["h", [], [target]],
            ["ms", [], [control, target]],
            ["h", [], [control]],
            ["gpi", [0.0], [control]],
            ["rz", [math.pi / 2], [control]],
            ["h", [], [target]],
            ["rz", [math.pi / 2], [target]],
            ["h", [], [target]],
        ]
