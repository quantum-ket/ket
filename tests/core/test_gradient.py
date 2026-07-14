# SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for parameter-shift gradient computation in parameterised quantum circuits.

libket implements the parameter-shift rule analytically in Rust via
``GradientStrategy::ParameterShiftRule``.  For a rotation gate
``G(θ) = exp(-iθP/2)`` (P a Pauli generator), the exact rule is:

    d⟨H⟩/dθ = ½ · [⟨H⟩(θ + π/2) - ⟨H⟩(θ - π/2)]

When the gate angle encodes a scaled parameter ``s·θ`` (multiplier *s*),
the chain rule propagates through:

    d⟨H⟩/dθ = s · ½ · [⟨H⟩(s·θ + π/2) - ⟨H⟩(s·θ - π/2)]

Every "matches_ps" test rebuilds the same circuit twice (θ ± π/2), computes
the shift manually with raw floats, and compares against ``t.grad``.
This avoids any reliance on noisy finite-difference approximations.
"""

from math import pi, cos, sin
import pytest
import ket
from ket import Process, RX, RY, RZ, exp_value
from ket.gates import obs


# ─── Reference implementations ──────────────────────────────────────────────


def _ps_gradient(circuit_fn, theta: float, multiplier: float = 1.0) -> float:
    """Exact parameter-shift estimate for a gate with angle ``multiplier * theta``.

    Shifts the *gate angle* (``multiplier * theta``) by ±π/2, then applies the
    chain rule to recover d⟨H⟩/dθ:

        d⟨H⟩/dθ = multiplier · [f(mult·θ + π/2) - f(mult·θ - π/2)] / 2
    """
    angle = multiplier * theta
    return multiplier * (circuit_fn(angle + pi / 2) - circuit_fn(angle - pi / 2)) / 2.0


# ─── Single-gate, single-qubit tests ─────────────────────────────────────────


class TestRXGradientZ:
    """RX(θ)|0⟩ measured with ⟨Z⟩.

    Analytically: ⟨Z⟩ = cos θ,  d⟨Z⟩/dθ = -sin θ.
    """

    def _ket_grad(self, theta: float):
        """Return (⟨Z⟩, dE/dθ) via ket's built-in parameter-shift."""
        p = Process(gradient=True)
        t = p.param(theta)
        q = p.alloc()
        RX(t, q)
        with obs():
            h = ket.Z(q)
        ev = exp_value(h)
        ev.get()
        return ev.value, t.grad

    def _ev(self, angle: float) -> float:
        """Expectation value for a given *gate angle* (raw float, no Parameter)."""
        p = Process()
        q = p.alloc()
        RX(angle, q)
        with obs():
            h = ket.Z(q)
        return exp_value(h).get()

    @pytest.mark.parametrize("theta", [0.0, pi / 6, pi / 4, pi / 3, pi / 2, pi, 1.234])
    def test_gradient_matches_ps(self, theta):
        """ket gradient equals an independent parameter-shift evaluation."""
        _, grad = self._ket_grad(theta)
        expected = _ps_gradient(self._ev, theta)
        assert grad == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("theta", [0.0, pi / 4, pi / 2, pi])
    def test_gradient_matches_analytical(self, theta):
        """d⟨Z⟩/dθ = -sin θ."""
        _, grad = self._ket_grad(theta)
        assert grad == pytest.approx(-sin(theta), abs=1e-6)

    @pytest.mark.parametrize("theta", [0.0, pi / 4, pi / 3, pi / 2])
    def test_expectation_value_is_correct(self, theta):
        """⟨Z⟩ = cos θ."""
        ev, _ = self._ket_grad(theta)
        assert ev == pytest.approx(cos(theta), abs=1e-6)


class TestRYGradientZ:
    """RY(θ)|0⟩ measured with ⟨Z⟩.

    Analytically: ⟨Z⟩ = cos θ,  d⟨Z⟩/dθ = -sin θ.
    """

    def _ket_grad(self, theta: float):
        p = Process(gradient=True)
        t = p.param(theta)
        q = p.alloc()
        RY(t, q)
        with obs():
            h = ket.Z(q)
        ev = exp_value(h)
        ev.get()
        return ev.value, t.grad

    def _ev(self, angle: float) -> float:
        p = Process()
        q = p.alloc()
        RY(angle, q)
        with obs():
            h = ket.Z(q)
        return exp_value(h).get()

    @pytest.mark.parametrize("theta", [0.0, pi / 6, pi / 4, pi / 3, pi / 2, pi, 2.0])
    def test_gradient_matches_ps(self, theta):
        _, grad = self._ket_grad(theta)
        expected = _ps_gradient(self._ev, theta)
        assert grad == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("theta", [0.0, pi / 4, pi / 2, pi])
    def test_gradient_matches_analytical(self, theta):
        """d⟨Z⟩/dθ = -sin θ for RY(θ)|0⟩."""
        _, grad = self._ket_grad(theta)
        assert grad == pytest.approx(-sin(theta), abs=1e-6)


class TestRZGradientX:
    """H|0⟩ → RZ(θ) measured with ⟨X⟩.

    Analytically: ⟨X⟩ = cos θ,  d⟨X⟩/dθ = -sin θ.
    """

    def _ket_grad(self, theta: float):
        p = Process(gradient=True)
        t = p.param(theta)
        q = p.alloc()
        ket.H(q)
        RZ(t, q)
        with obs():
            h = ket.X(q)
        ev = exp_value(h)
        ev.get()
        return ev.value, t.grad

    def _ev(self, angle: float) -> float:
        p = Process()
        q = p.alloc()
        ket.H(q)
        RZ(angle, q)
        with obs():
            h = ket.X(q)
        return exp_value(h).get()

    @pytest.mark.parametrize("theta", [0.0, pi / 6, pi / 4, pi / 3, pi / 2, pi, 1.5])
    def test_gradient_matches_ps(self, theta):
        _, grad = self._ket_grad(theta)
        expected = _ps_gradient(self._ev, theta)
        assert grad == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("theta", [0.0, pi / 4, pi / 2, pi])
    def test_gradient_matches_analytical(self, theta):
        """d⟨X⟩/dθ = -sin θ for H·RZ(θ)|0⟩."""
        _, grad = self._ket_grad(theta)
        assert grad == pytest.approx(-sin(theta), abs=1e-6)


# ─── Multi-parameter tests ────────────────────────────────────────────────────


class TestMultiParameterGradient:
    """Two independent parameters θ and φ in the same circuit.

    Circuit: RX(θ)·RY(φ)|0⟩, observable ⟨Z⟩.
    """

    def _ket_grad(self, theta: float, phi: float):
        p = Process(gradient=True)
        t, ph = p.param(theta, phi)
        q = p.alloc()
        RX(t, q)
        RY(ph, q)
        with obs():
            h = ket.Z(q)
        ev = exp_value(h)
        ev.get()
        return ev.value, t.grad, ph.grad

    def _ev(self, angle1: float, angle2: float) -> float:
        p = Process()
        q = p.alloc()
        RX(angle1, q)
        RY(angle2, q)
        with obs():
            h = ket.Z(q)
        return exp_value(h).get()

    @pytest.mark.parametrize(
        "theta,phi",
        [(pi / 4, pi / 6), (pi / 3, pi / 4), (pi / 2, pi / 3), (0.5, 1.0)],
    )
    def test_gradient_theta_matches_ps(self, theta, phi):
        """dE/dθ via ket equals independent PS estimate with φ fixed."""
        _, grad_t, _ = self._ket_grad(theta, phi)
        expected = _ps_gradient(lambda a: self._ev(a, phi), theta)
        assert grad_t == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize(
        "theta,phi",
        [(pi / 4, pi / 6), (pi / 3, pi / 4), (pi / 2, pi / 3)],
    )
    def test_gradient_phi_matches_ps(self, theta, phi):
        """dE/dφ via ket equals independent PS estimate with θ fixed."""
        _, _, grad_ph = self._ket_grad(theta, phi)
        expected = _ps_gradient(lambda a: self._ev(theta, a), phi)
        assert grad_ph == pytest.approx(expected, abs=1e-6)


# ─── Scaled parameter tests ───────────────────────────────────────────────────


class TestScaledParameter:
    """Gradient when a Parameter is multiplied by a scalar.

    Circuit: RX(scale·θ)|0⟩, observable ⟨Z⟩.
    Analytically: d⟨Z⟩/dθ = -sin(scale·θ) · scale  (chain rule).
    """

    def _ket_grad(self, theta: float, scale: float) -> float:
        p = Process(gradient=True)
        t = p.param(theta)
        q = p.alloc()
        RX(t * scale, q)
        with obs():
            h = ket.Z(q)
        ev = exp_value(h)
        ev.get()
        return t.grad

    def _ev(self, angle: float) -> float:
        p = Process()
        q = p.alloc()
        RX(angle, q)
        with obs():
            h = ket.Z(q)
        return exp_value(h).get()

    @pytest.mark.parametrize(
        "theta,scale", [(pi / 4, 2.0), (pi / 6, 0.5), (1.0, 3.0)]
    )
    def test_scaled_gradient_chain_rule(self, theta, scale):
        """d⟨Z⟩/dθ = -sin(scale·θ) · scale (chain rule)."""
        grad = self._ket_grad(theta, scale)
        assert grad == pytest.approx(-sin(scale * theta) * scale, abs=1e-6)

    @pytest.mark.parametrize(
        "theta,scale", [(pi / 4, 2.0), (pi / 6, 0.5), (1.0, 3.0)]
    )
    def test_scaled_gradient_matches_ps(self, theta, scale):
        """Gradient w.r.t. raw θ equals the PS estimate accounting for the multiplier."""
        grad = self._ket_grad(theta, scale)
        expected = _ps_gradient(self._ev, theta, multiplier=scale)
        assert grad == pytest.approx(expected, abs=1e-6)

    def test_negated_parameter(self):
        """Using -θ as gate angle: d⟨Z⟩/dθ = -sin(θ).

        Gate angle = -1 · θ, so ⟨Z⟩ = cos(-θ) = cos(θ).
        Chain rule: d cos(-θ)/dθ = -sin(θ).
        """
        theta = pi / 4
        p = Process(gradient=True)
        t = p.param(theta)
        q = p.alloc()
        RX(-t, q)
        with obs():
            h = ket.Z(q)
        exp_value(h).get()
        # ⟨Z⟩ = cos(θ) regardless of the sign flip, derivative is -sin(θ)
        assert t.grad == pytest.approx(-sin(theta), abs=1e-6)

    def test_divided_parameter(self):
        """Using θ/2 as gate angle: d⟨Z⟩/dθ = -sin(θ/2)/2."""
        theta = pi / 3
        p = Process(gradient=True)
        t = p.param(theta)
        q = p.alloc()
        RX(t / 2, q)
        with obs():
            h = ket.Z(q)
        exp_value(h).get()
        assert t.grad == pytest.approx(-sin(theta / 2) / 2, abs=1e-6)


# ─── Parameter reuse across multiple gates ─────────────────────────────────


class TestParameterReuseGradient:
    """Same Parameter object used in multiple gates — gradient accumulates.

    Circuit: RX(θ)·RY(θ) on one qubit, observable ⟨Z⟩.
    Total gradient dE/dθ is the sum of the PS shifts over each gate.
    """

    def _ket_grad(self, theta: float):
        p = Process(gradient=True)
        t = p.param(theta)
        q = p.alloc()
        RX(t, q)
        RY(t, q)
        with obs():
            h = ket.Z(q)
        ev = exp_value(h)
        ev.get()
        return ev.value, t.grad

    def _ev_rx_shift_ry_fixed(self, rx_angle: float, ry_angle: float) -> float:
        """Circuit with independently controlled RX and RY angles."""
        p = Process()
        q = p.alloc()
        RX(rx_angle, q)
        RY(ry_angle, q)
        with obs():
            h = ket.Z(q)
        return exp_value(h).get()

    @pytest.mark.parametrize("theta", [0.1, pi / 6, pi / 4, pi / 3, 0.9])
    def test_gradient_equals_sum_of_ps_contributions(self, theta):
        """Total gradient = PS contribution from RX + PS contribution from RY."""
        _, grad = self._ket_grad(theta)

        # Shift only the RX gate (RY stays at theta)
        ps_rx = (
            self._ev_rx_shift_ry_fixed(theta + pi / 2, theta)
            - self._ev_rx_shift_ry_fixed(theta - pi / 2, theta)
        ) / 2.0

        # Shift only the RY gate (RX stays at theta)
        ps_ry = (
            self._ev_rx_shift_ry_fixed(theta, theta + pi / 2)
            - self._ev_rx_shift_ry_fixed(theta, theta - pi / 2)
        ) / 2.0

        expected = ps_rx + ps_ry
        assert grad == pytest.approx(expected, abs=1e-6)


# ─── Gradient availability ────────────────────────────────────────────────────


class TestGradientAvailability:
    """grad should be None when ``gradient=False``, and a float when ``gradient=True``."""

    def test_grad_is_none_without_flag(self):
        p = Process(gradient=False)
        t = p.param(pi / 4)
        q = p.alloc()
        RX(t, q)
        with obs():
            h = ket.Z(q)
        exp_value(h).get()
        assert t.grad is None

    def test_grad_available_with_flag(self):
        p = Process(gradient=True)
        t = p.param(pi / 4)
        q = p.alloc()
        RX(t, q)
        with obs():
            h = ket.Z(q)
        exp_value(h).get()
        assert t.grad is not None
        assert isinstance(t.grad, float)


# ─── Two-qubit entangled circuits ─────────────────────────────────────────────


class TestEntangledCircuitGradient:
    """Gradient in circuits with entanglement.

    Circuit: H|0⟩, CNOT, RX(θ) on target, observable ⟨ZZ⟩.
    The only parametrised gate is RX(θ) on qubit 1; PS still applies.
    """

    def _ket_grad(self, theta: float):
        p = Process(gradient=True)
        t = p.param(theta)
        q = p.alloc(2)
        ket.H(q[0])
        ket.CNOT(q[0], q[1])
        RX(t, q[1])
        with obs():
            h = ket.Z(q[0]) * ket.Z(q[1])
        ev = exp_value(h)
        ev.get()
        return ev.value, t.grad

    def _ev(self, angle: float) -> float:
        p = Process()
        q = p.alloc(2)
        ket.H(q[0])
        ket.CNOT(q[0], q[1])
        RX(angle, q[1])
        with obs():
            h = ket.Z(q[0]) * ket.Z(q[1])
        return exp_value(h).get()

    @pytest.mark.parametrize("theta", [0.0, pi / 6, pi / 4, pi / 3, pi / 2, 1.5])
    def test_gradient_matches_ps(self, theta):
        """ket gradient equals independent PS evaluation for the entangled circuit."""
        _, grad = self._ket_grad(theta)
        expected = _ps_gradient(self._ev, theta)
        assert grad == pytest.approx(expected, abs=1e-6)


# ─── Hamiltonian with multiple Pauli terms ────────────────────────────────────


class TestMultiTermHamiltonianGradient:
    """Gradient w.r.t. a Hamiltonian with multiple Pauli terms.

    Circuit: RY(θ)|0⟩, observable ⟨X + Z⟩.
    Analytically: d(sin θ + cos θ)/dθ = cos θ - sin θ.
    """

    def _ket_grad(self, theta: float):
        p = Process(gradient=True)
        t = p.param(theta)
        q = p.alloc()
        RY(t, q)
        with obs():
            h = ket.X(q) + ket.Z(q)
        ev = exp_value(h)
        ev.get()
        return ev.value, t.grad

    def _ev(self, angle: float) -> float:
        p = Process()
        q = p.alloc()
        RY(angle, q)
        with obs():
            h = ket.X(q) + ket.Z(q)
        return exp_value(h).get()

    @pytest.mark.parametrize("theta", [0.0, pi / 6, pi / 4, pi / 3, pi / 2, 1.0])
    def test_gradient_matches_ps(self, theta):
        """ket gradient equals independent PS evaluation."""
        _, grad = self._ket_grad(theta)
        expected = _ps_gradient(self._ev, theta)
        assert grad == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("theta", [0.0, pi / 4, pi / 2])
    def test_gradient_matches_analytical(self, theta):
        """d(sin θ + cos θ)/dθ = cos θ - sin θ."""
        _, grad = self._ket_grad(theta)
        assert grad == pytest.approx(cos(theta) - sin(theta), abs=1e-6)


# ─── Batch execution mode ──────────────────────────────────────────────────────


class TestGradientBatchMode:
    """Gradient computation works in explicit batch execution mode."""

    @pytest.mark.parametrize("theta", [pi / 4, pi / 3, pi / 2])
    def test_rx_gradient_batch(self, theta):
        p = Process(execution="batch", gradient=True)
        t = p.param(theta)
        q = p.alloc()
        RX(t, q)
        with obs():
            h = ket.Z(q)
        exp_value(h).get()
        assert t.grad == pytest.approx(-sin(theta), abs=1e-6)

    @pytest.mark.parametrize("theta", [pi / 4, pi / 3, pi / 2])
    def test_ry_gradient_batch(self, theta):
        p = Process(execution="batch", gradient=True)
        t = p.param(theta)
        q = p.alloc()
        RY(t, q)
        with obs():
            h = ket.Z(q)
        exp_value(h).get()
        assert t.grad == pytest.approx(-sin(theta), abs=1e-6)


# ─── Zero-gradient and extremum cases ─────────────────────────────────────────


class TestZeroAndExtremumGradient:
    """Verify zero gradient and maximum-magnitude gradient at known stationary points."""

    def test_rx_gradient_zero_at_zero(self):
        """d⟨Z⟩/dθ = -sin(0) = 0."""
        p = Process(gradient=True)
        t = p.param(0.0)
        q = p.alloc()
        RX(t, q)
        with obs():
            h = ket.Z(q)
        exp_value(h).get()
        assert t.grad == pytest.approx(0.0, abs=1e-6)

    def test_rx_gradient_zero_at_pi(self):
        """d⟨Z⟩/dθ = -sin(π) ≈ 0."""
        p = Process(gradient=True)
        t = p.param(pi)
        q = p.alloc()
        RX(t, q)
        with obs():
            h = ket.Z(q)
        exp_value(h).get()
        assert t.grad == pytest.approx(0.0, abs=1e-6)

    def test_rx_gradient_max_at_pi_over_2(self):
        """d⟨Z⟩/dθ = -sin(π/2) = -1 (maximum magnitude)."""
        p = Process(gradient=True)
        t = p.param(pi / 2)
        q = p.alloc()
        RX(t, q)
        with obs():
            h = ket.Z(q)
        exp_value(h).get()
        assert t.grad == pytest.approx(-1.0, abs=1e-6)

    def test_ry_gradient_max_at_pi_over_2(self):
        """d⟨Z⟩/dθ = -sin(π/2) = -1 for RY."""
        p = Process(gradient=True)
        t = p.param(pi / 2)
        q = p.alloc()
        RY(t, q)
        with obs():
            h = ket.Z(q)
        exp_value(h).get()
        assert t.grad == pytest.approx(-1.0, abs=1e-6)


# ─── Parameter value and scaling properties ────────────────────────────────────


class TestParameterValueProperties:
    """Validate Parameter arithmetic and the .value / .param properties."""

    def test_param_value_is_initial(self):
        p = Process(gradient=True)
        theta = pi / 3
        t = p.param(theta)
        assert t.value == pytest.approx(theta)
        assert t.param == pytest.approx(theta)

    def test_scaled_param_value(self):
        p = Process(gradient=True)
        t = p.param(pi / 3)
        scaled = t * 2
        assert scaled.value == pytest.approx((pi / 3) * 2)

    def test_divided_param_value(self):
        p = Process(gradient=True)
        t = p.param(pi / 3)
        divided = t / 2
        assert divided.value == pytest.approx((pi / 3) / 2)

    def test_negated_param_value(self):
        p = Process(gradient=True)
        t = p.param(pi / 3)
        negated = -t
        assert negated.value == pytest.approx(-(pi / 3))
