"""Quantum state snapshot.

This module provides the base class for a snapshot of a quantum state obtained
using a quantum simulator.
"""

from __future__ import annotations

# SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

from math import pi
from random import Random
from cmath import sqrt, phase
from collections import defaultdict
from typing import Literal, Callable
from ctypes import c_size_t

from .base import Quant, _check_visualize

try:
    import numpy as np
    import plotly.graph_objs as go
    import plotly.express as px
except ImportError:
    pass

try:
    from IPython import get_ipython
    from IPython.display import Math

    try:
        import google.colab  # pylint: disable=unused-import

        _IN_NOTEBOOK = True
    except ImportError:
        _IN_NOTEBOOK = get_ipython().__class__.__name__ == "ZMQInteractiveShell"
except ImportError:
    _IN_NOTEBOOK = False

__all__ = ["QuantumState"]


class QuantumState:
    """Snapshot of a quantum state.

    This class holds a reference for a snapshot of a quantum state obtained using a quantum
    simulator. The result may not be available right after the measurement call, especially in batch
    execution.

    You can instantiate this class by calling the :func:`~ket.operations.dump` function.

    Args:
        qubits: Qubits from which to capture a quantum state snapshot.
    """

    def __init__(self, *qubits: Quant):
        self.qubits = []
        self.qubits_info: list[tuple[int, Callable[[int], str]]] = []

        for qubit in qubits:
            self.qubits.extend(qubit.qubits)
            self.qubits_info.append((len(qubit), qubit.dump_format()))

        self.process = qubits[0].process if qubits else None

        if self.process:
            self.index = self.process.dump(
                (c_size_t * len(self.qubits))(*self.qubits), len(self.qubits)
            ).value

        self.size = len(self.qubits)
        self._states = None

    def _get_ket_process(self):
        return self.process

    def _check(self):
        if self._states is None:
            available, size = self.process.get_dump_size(self.index)
            if available.value:
                states = defaultdict(complex)
                for i in range(size.value):
                    state, state_size, amp_real, amp_imag = self.process.get_dump(
                        self.index, i
                    )
                    state = int(
                        "".join(f"{state[j]:064b}" for j in range(state_size.value)), 2
                    )
                    amplitude = complex(amp_real.value, amp_imag.value)
                    if abs(amplitude) ** 2 > 1e-10:
                        states[state] += amplitude

                p = sum(map(lambda a: abs(a) ** 2, states.values()))
                if abs(p - 1.0) < 1e-10:
                    self._states = dict(states)
                else:
                    self._states = {
                        state: amplitude / sqrt(p) if p != 0 else 0
                        for state, amplitude in states.items()
                        if abs(amplitude) ** 2 > 1e-10
                    }

    @property
    def states(self) -> dict[int, complex] | None:
        """Get the quantum state.

        Returns a dictionary mapping base states to their corresponding probability amplitudes.

        Returns:
            The quantum state, or None if the quantum state information is not available.
        """
        self._check()
        return self._states

    def get(self) -> dict[int, complex]:
        """Get the quantum state.

        If the quantum state is not available, the quantum process will execute to get the result.
        """
        self._check()
        if self._states is None:
            self.process.execute()
        return self.states

    @property
    def probabilities(self) -> dict[int, float] | None:
        """Get the measurement probabilities.

        Returns a dictionary mapping base states to their corresponding measurement probabilities.

        Returns:
            The measurement probabilities, or None if the quantum state information is not
            available.
        """
        self._check()
        if self._states is None:
            return None
        return dict(map(lambda a: (a[0], abs(a[1]) ** 2), self._states.items()))

    def sample(self, shots=4096, seed=None) -> dict[int, int] | None:
        """Get the quantum execution shots.

        The parameters ``shots`` and ``seed`` are used to generate the sample from the quantum state
        snapshot.

        Args:
            shots: Number of shots. Defaults to 4096.
            seed: Seed for the RNG.

        Returns:
            A dictionary mapping measurement outcomes to their frequencies in the generated sample,
            or None if the quantum state information is not available.
        """
        self._check()
        if self._states is None:
            return None

        rng = Random(seed)
        shots = rng.choices(list(self.states), list(self.probabilities), k=shots)
        result = {}
        for state in shots:
            if state not in result:
                result[state] = 1
            else:
                result[state] += 1
        return result

    @staticmethod
    def _sphere():  # pylint: disable=too-many-locals
        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2 * np.pi, 40)
        phi, theta = np.meshgrid(phi, theta)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        sphere = go.Surface(
            x=x, y=y, z=z, showscale=False, opacity=0.02, name="Bloch Sphere"
        )

        equator_theta = np.linspace(0, 2 * np.pi, 100)
        equator_x = np.cos(equator_theta)
        equator_y = np.sin(equator_theta)
        equator_z = np.zeros_like(equator_theta)

        equator = go.Scatter3d(
            x=equator_x,
            y=equator_y,
            z=equator_z,
            mode="lines",
            line={"color": "gray", "width": 3},
            opacity=0.1,
            name="Equator",
        )

        z_line = go.Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[1, -1],
            mode="lines",
            line={"color": "gray", "width": 3},
            opacity=0.1,
            name="z line",
        )

        x_line = go.Scatter3d(
            x=[1, -1],
            y=[0, 0],
            z=[0, 0],
            mode="lines",
            line={"color": "gray", "width": 3},
            opacity=0.1,
            name="x line",
        )

        y_line = go.Scatter3d(
            x=[0, 0],
            y=[1, -1],
            z=[0, 0],
            mode="lines",
            line={"color": "gray", "width": 3},
            opacity=0.1,
            name="y line",
        )

        basis_points = [
            ([0.0, 0.0, 1.0], "|0⟩"),
            ([0.0, 0.0, 0.8], "Z"),
            ([0.0, 0.0, -1.0], "|1⟩"),
            ([1.0, 0.0, 0.0], "|+⟩"),
            ([0.8, 0.0, 0.0], "X"),
            ([-1.0, 0.0, 0.0], "|‒⟩"),
            ([0.0, 1.0, 0.0], "|+i⟩"),
            ([0.0, 0.8, 0.0], "Y"),
            ([0.0, -1.0, 0.0], "|-i⟩"),
        ]

        basis = [
            go.Scatter3d(
                x=[p[0]],
                y=[p[1]],
                z=[p[2]],
                mode="text",
                text=[text],
                textposition="middle center",
                name=text,
            )
            for p, text in basis_points
        ]

        return [
            sphere,
            equator,
            x_line,
            y_line,
            z_line,
            *basis,
        ]

    def sphere(self) -> go.Figure:
        """Generate a Bloch sphere plot representing the quantum state.

        This method creates a Bloch sphere plot visualizing of one qubit quantum state.

        Note:
            This method requires additional dependencies from ``ket-lang[plot]``.

            Install with: ``pip install ket-lang[plot]``.

        Returns:
            A Bloch sphere plot illustrating the quantum state.
        """
        if len(self.qubits) != 1:
            raise ValueError("Bloch sphere plot is available only for 1 qubit")
        _check_visualize()

        ket = np.array(
            [
                [self.get()[0] if 0 in self.get() else 0.0],
                [self.get()[1] if 1 in self.get() else 0.0],
            ]
        )

        bra = np.conjugate(ket.T)

        pauli_x = np.array([[0, 1], [1, 0]])
        pauli_y = np.array([[0, -1j], [1j, 0]])
        pauli_z = np.array([[1, 0], [0, -1]])
        exp_x = (bra @ pauli_x @ ket).item().real
        exp_y = (bra @ pauli_y @ ket).item().real
        exp_z = (bra @ pauli_z @ ket).item().real

        qubit = go.Scatter3d(
            x=[exp_x],
            y=[exp_y],
            z=[exp_z],
            mode="markers",
            marker={"size": 5, "color": "red"},
            name="qubit",
        )

        line = go.Scatter3d(
            x=[0, exp_x],
            y=[0, exp_y],
            z=[0, exp_z],
            mode="lines",
            line={"color": "red", "width": 3},
            opacity=0.5,
            name="qubit line",
        )

        fig = go.Figure(
            data=[
                *self._sphere(),
                qubit,
                line,
            ]
        )

        fig.update_layout(
            scene={
                "xaxis": {
                    "range": [-1, 1],
                    "showgrid": False,
                    "showbackground": False,
                    "visible": False,
                },
                "yaxis": {
                    "range": [-1, 1],
                    "showgrid": False,
                    "showbackground": False,
                    "visible": False,
                },
                "zaxis": {
                    "range": [-1, 1],
                    "showgrid": False,
                    "showbackground": False,
                    "visible": False,
                },
                "aspectmode": "cube",
                "camera": {
                    "eye": {
                        "x": 0.8,
                        "y": 0.8,
                        "z": 0.8,
                    },
                },
            },
            showlegend=False,
        )

        return fig

    def show(
        self,
        mode: Literal["latex", "str"] | None = None,
        round_tol: float = 1e-6,
    ) -> str:
        r"""Return the quantum state as a formatted string or LaTeX Math object.

        Args:
            mode: If ``"str"``, returns a plain text string. If ``"latex"``, returns a LaTeX
                Math representation. Defaults to ``"latex"`` in Jupyter Notebooks,
                otherwise ``"str"``.
            round_tol: Numerical tolerance used when rounding values.

        Returns:
            The formatted quantum state as a string, or Latex Math.
        """
        if mode is None:
            mode = "latex" if _IN_NOTEBOOK else "str"
        elif mode not in ("latex", "str"):
            raise ValueError(f"Unknown mode: {mode}")

        if mode == "latex":
            return self._show_latex(round_tol)
        return self._show_str(round_tol)

    def _get_ket_parts(self, state_bin: str, latex: bool = False) -> list[str]:
        """Helper to extract, slice and format the inner value of a ket state."""
        parts = []
        current = 0
        for size, formatter in self.qubits_info:
            slice_bin = state_bin[current : current + size]
            val = int(slice_bin, 2) if slice_bin else 0
            formatted_val = formatter(val)

            if latex:
                parts.append(f"\\left|{formatted_val}\\right>")
            else:
                parts.append(f"|{formatted_val}⟩")

            current += size
        return parts

    def _show_str(self, round_tol: float = 1e-6) -> str:

        def state_amp_str(state, amp):
            if abs(amp) < round_tol:
                return ""

            state_bin = f"{state:0{self.size}b}"

            ket_parts = self._get_ket_parts(state_bin, latex=False)
            dump_str = "".join(ket_parts)

            dump_str += f"\t({100*abs(amp)**2:.2f}%)\n"

            real = abs(amp.real) > round_tol
            real_l0 = amp.real < 0

            imag = abs(amp.imag) > round_tol
            imag_l0 = amp.imag < 0

            amp_sq = abs(amp) ** 2
            sqrt_dem_val = 1 / amp_sq

            use_sqrt = abs(round(sqrt_dem_val) - sqrt_dem_val) < round_tol
            use_sqrt = use_sqrt and (
                (abs(abs(amp.real) - abs(amp.imag)) < round_tol) or (real != imag)
            )

            sqrt_dem = f"/√{round(sqrt_dem_val)}"

            if real and imag:
                sqrt_dem = f"/√{round(2 * sqrt_dem_val)}"
                sqrt_num = ("(-1" if real_l0 else " (1") + ("-i" if imag_l0 else "+i")

                sqrt_str = (
                    f"\t≅ {sqrt_num}){sqrt_dem}"
                    if use_sqrt and (abs(abs(amp.real) - abs(amp.imag)) < round_tol)
                    else ""
                )
                dump_str += f"{amp.real:9.6f}{amp.imag:+.6f}i" + sqrt_str

            elif real:
                sqrt_num = "  -1" if real_l0 else "   1"
                sqrt_str = f"\t≅   {sqrt_num}{sqrt_dem}" if use_sqrt else ""
                dump_str += f"{amp.real:9.6f}       " + sqrt_str

            else:
                sqrt_num = "  -i" if imag_l0 else "   i"
                sqrt_str = f"\t≅   {sqrt_num}{sqrt_dem}" if use_sqrt else ""
                dump_str += f" {amp.imag:17.6f}i" + sqrt_str

            return dump_str

        lines = [
            state_amp_str(state, amp)
            for state, amp in sorted(self.get().items(), key=lambda k: k[0])
        ]
        return "\n".join(line for line in lines if line)

    def _show_latex(self, round_tol: float = 1e-6) -> Math:

        def float_to_math(num: float, is_complex: bool) -> str | None:
            if abs(num) < round_tol:
                return None

            sqrt_dem_float = 1 / num**2
            sqrt_dem = round(sqrt_dem_float)
            if abs(sqrt_dem - sqrt_dem_float) < round_tol and sqrt_dem != 1:
                sign = "-" if num < 0.0 else ""
                numerator = "i" if is_complex else "1"
                return f"\\frac{{{sign}{numerator}}}{{\\sqrt{{{sqrt_dem}}}}}"

            round_num = round(num)
            if abs(round_num - num) < round_tol:
                if round_num == 1:
                    num_str = ""
                elif round_num == -1:
                    num_str = "-"
                else:
                    num_str = str(round_num)
            else:
                num_str = str(num)
                if "e" in num_str:
                    num_str = num_str.replace("e", "\\times10^{") + "}"

            if is_complex:
                num_str += "i"
                if num_str in ("i", "-i"):
                    pass

            return num_str

        math_terms = []
        for state, amp in self.get().items():
            if abs(amp) < round_tol:
                continue

            real_str = float_to_math(amp.real, False)
            imag_str = float_to_math(amp.imag, True)

            state_bin = f"{state:0{len(self.qubits)}b}"

            state_parts = self._get_ket_parts(state_bin, latex=True)
            state_str_joined = "".join(state_parts)

            if real_str is not None and imag_str is not None:
                math_terms.append(f"({real_str}+{imag_str}){state_str_joined}")
            else:
                coeff = real_str if real_str is not None else imag_str
                math_terms.append(f"{coeff}{state_str_joined}")

        # Clean up any "+-" instances created during formatting
        return Math("+".join(math_terms).replace("+-", "-"))

    def histogram(self, mode: Literal["bin", "dec"] = "dec", **kwargs) -> go.Figure:
        """Generate a histogram representing the quantum state.

        This method creates a histogram visualizing the probability distribution
        of the quantum state.

        Note:
            This method requires additional dependencies from ``ket-lang[plot]``.

            Install with: ``pip install ket-lang[plot]``.

        Args:
            mode: If ``"bin"``, display the states in binary format. If ``"dec"``,
            display the states in decimal format. Defaults to ``"dec"``.
            **kwargs: Additional keyword arguments passed to :func:`plotly.express.bar`.

        Returns:
            Histogram of the quantum state.
        """
        _check_visualize()

        state = list(self.get().keys())
        state_text = (
            list(map(lambda s: f"|{s:0{self.size}b}⟩", state))
            if mode == "bin"
            else state
        )

        data = {
            "State": state,
            "Probability": list(map(lambda a: abs(a) ** 2, self.get().values())),
            "Phase": list(map(phase, self.get().values())),
        }

        fig = px.bar(
            data,
            x="State",
            y="Probability",
            color="Phase",
            range_color=(-pi, pi),
            **kwargs,
        )

        fig.update_layout(
            xaxis={
                "tickmode": "array",
                "ticktext": state_text,
                "tickvals": state,
            },
            bargap=0.75,
        )

        return fig

    def __repr__(self):
        return f"<Ket 'QuantumState' index={self.index}, pid={hex(id(self.process))}>"
