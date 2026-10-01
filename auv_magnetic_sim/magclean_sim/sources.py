from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .math3d import (
    dipole_linear_map_nt,
    gradient_stencil_points,
    stencil_fields_to_state8,
)


@dataclass
class MagneticSource:
    name: str
    position_m: np.ndarray
    moment_axis: np.ndarray
    gain_am2: float
    quadratic: float
    lag_s: float
    temperature_coefficient: float
    cross_coupling: float


def default_sources(
    position_overrides_m: Dict[str, List[float]] = None,
    strength_scale: float = 1.0,
) -> List[MagneticSource]:
    raw = [
        ("thruster", [-0.82, 0.10, -0.12], [0.92, 0.28, 0.18], 0.090, 0.18, 0.22, 0.0040, 0.08),
        ("dc_dc", [-0.38, -0.24, 0.13], [0.30, -0.90, 0.24], 0.045, 0.10, 0.08, 0.0060, -0.05),
        ("actuator", [0.27, 0.22, -0.17], [-0.42, 0.20, 0.88], 0.030, 0.22, 0.16, 0.0030, 0.06),
        ("electronics", [0.05, -0.31, 0.08], [0.58, 0.67, -0.46], 0.024, 0.06, 0.05, 0.0070, 0.04),
    ]
    sources = []
    position_overrides_m = position_overrides_m or {}
    for values in raw:
        axis = np.asarray(values[2], dtype=float)
        axis = axis / np.linalg.norm(axis)
        position = position_overrides_m.get(values[0], values[1])
        sources.append(
            MagneticSource(
                name=values[0],
                position_m=np.asarray(position, dtype=float),
                moment_axis=axis,
                gain_am2=values[3] * float(strength_scale),
                quadratic=values[4],
                lag_s=values[5],
                temperature_coefficient=values[6],
                cross_coupling=values[7],
            )
        )
    return sources


class MagneticSourceSystem:
    """Forward model for onboard magnetic dipole sources.

    It is intentionally more nonlinear and dynamic than the identified ridge
    model. This creates a meaningful, controlled model-mismatch experiment.
    """

    def __init__(
        self,
        drift_fraction: float = 0.0,
        sources: List[MagneticSource] = None,
        source_positions_m: Dict[str, List[float]] = None,
        strength_scale: float = 1.0,
    ):
        self.sources = sources or default_sources(source_positions_m, strength_scale)
        drift_pattern = np.array([0.8, -0.5, 1.0, -0.7])
        self.gain_scales = 1.0 + float(drift_fraction) * drift_pattern[: len(self.sources)]

    def _effective_states(self, states: np.ndarray, dt_s: float) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        effective = np.zeros_like(states)
        effective[0] = states[0]
        for index, source in enumerate(self.sources):
            alpha = np.exp(-dt_s / max(source.lag_s, 1.0e-4))
            for sample in range(1, len(states)):
                effective[sample, index] = (
                    alpha * effective[sample - 1, index]
                    + (1.0 - alpha) * states[sample, index]
                )
        return effective

    def magnetic_moments(
        self, states: np.ndarray, temperature_c: np.ndarray, dt_s: float
    ) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        temperature = np.asarray(temperature_c, dtype=float)
        effective = self._effective_states(states, dt_s)
        moments = np.zeros((states.shape[0], len(self.sources), 3), dtype=float)
        for index, source in enumerate(self.sources):
            neighbor = states[:, (index + 1) % states.shape[1]]
            amplitude = (
                effective[:, index]
                + source.quadratic * states[:, index] ** 2
                + source.cross_coupling * states[:, index] * neighbor
            )
            amplitude *= 1.0 + source.temperature_coefficient * (temperature - 20.0)
            amplitude *= source.gain_am2 * self.gain_scales[index]
            # A small state-dependent axis rotation mimics non-ideal wiring/structure coupling.
            secondary_axis = np.roll(source.moment_axis, 1)
            moments[:, index, :] = (
                amplitude[:, None] * source.moment_axis[None, :]
                + 0.07
                * source.gain_am2
                * states[:, index, None]
                * neighbor[:, None]
                * secondary_axis[None, :]
            )
        return moments

    def field_at_points(
        self,
        states: np.ndarray,
        temperature_c: np.ndarray,
        dt_s: float,
        evaluation_points_m: np.ndarray,
    ) -> np.ndarray:
        contributions = self.field_contributions_at_points(
            states, temperature_c, dt_s, evaluation_points_m
        )
        return contributions.sum(axis=1)

    def field_contributions_at_points(
        self,
        states: np.ndarray,
        temperature_c: np.ndarray,
        dt_s: float,
        evaluation_points_m: np.ndarray,
    ) -> np.ndarray:
        """Return each source contribution as [sample, source, point, xyz] in nT."""
        moments = self.magnetic_moments(states, temperature_c, dt_s)
        points = np.asarray(evaluation_points_m, dtype=float)
        contributions = np.zeros(
            (states.shape[0], len(self.sources), len(points), 3), dtype=float
        )
        for index, source in enumerate(self.sources):
            mapping = dipole_linear_map_nt(source.position_m, points)
            contributions[:, index] = np.einsum(
                "pij,nj->npi", mapping, moments[:, index, :]
            )
        return contributions

    def field_and_gradient_state(
        self,
        states: np.ndarray,
        temperature_c: np.ndarray,
        dt_s: float,
        gradient_step_m: float = 0.01,
    ) -> np.ndarray:
        points = gradient_stencil_points(gradient_step_m)
        fields = self.field_at_points(states, temperature_c, dt_s, points)
        return stencil_fields_to_state8(fields, gradient_step_m)


def generate_scenario_states(
    scenario: str, sample_count: int, sample_rate_hz: float, rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate deterministic excitation families for scenario-disjoint calibration."""
    t = np.arange(sample_count, dtype=float) / sample_rate_hz
    states = np.zeros((sample_count, 4), dtype=float)
    base = np.array([0.12, 0.18, 0.05, 0.35])
    states[:] = base
    phase = rng.uniform(0.0, 2.0 * np.pi, size=4)
    slow_temperature = 20.0 + 3.0 * np.sin(2.0 * np.pi * t / max(t[-1] + 1.0, 10.0))

    def sweep(channel: int, low: float, high: float, cycles: float = 2.0) -> None:
        wave = 0.5 + 0.5 * np.sin(2.0 * np.pi * cycles * t / max(t[-1], 1.0) + phase[channel])
        steps = ((np.arange(sample_count) // max(sample_count // 6, 1)) % 3) / 2.0
        states[:, channel] = low + (high - low) * (0.65 * wave + 0.35 * steps)

    if scenario == "thruster_sweep":
        sweep(0, 0.05, 1.0, 2.5)
    elif scenario == "dc_dc_steps":
        sweep(1, 0.05, 0.95, 1.5)
    elif scenario == "actuator_cycle":
        states[:, 2] = 0.75 * np.sin(2.0 * np.pi * 1.8 * t / max(t[-1], 1.0))
    elif scenario == "electronics_cycle":
        sweep(3, 0.15, 1.0, 3.0)
    elif scenario == "thruster_dc_dc":
        sweep(0, 0.15, 1.0, 2.0)
        sweep(1, 0.10, 0.90, 1.3)
    elif scenario == "thruster_actuator":
        sweep(0, 0.10, 0.95, 2.2)
        states[:, 2] = 0.65 * np.sin(2.0 * np.pi * 1.4 * t / max(t[-1], 1.0) + 0.4)
    elif scenario == "dc_electronics_train":
        sweep(1, 0.04, 1.02, 2.7)
        sweep(3, 0.10, 1.03, 1.1)
    elif scenario == "actuator_electronics_train":
        states[:, 2] = 0.92 * np.sin(2.0 * np.pi * 2.6 * t / max(t[-1], 1.0) + 0.7)
        sweep(3, 0.08, 1.02, 1.8)
    elif scenario == "random_multisine_train":
        for channel in range(4):
            wave = (
                0.42 * np.sin(2.0 * np.pi * (0.55 + 0.18 * channel) * t + phase[channel])
                + 0.23 * np.sin(2.0 * np.pi * (1.35 + 0.11 * channel) * t + 0.4)
                + 0.12 * np.sin(2.0 * np.pi * (2.55 + 0.17 * channel) * t + 1.0)
            )
            states[:, channel] = (0.48 + wave) if channel != 2 else wave
    elif scenario == "thermal_mixed_train":
        for channel in range(4):
            sweep(channel, -0.85 if channel == 2 else 0.03, 1.04, 1.2 + 0.42 * channel)
        slow_temperature = 14.0 + 16.0 * t / max(t[-1], 1.0)
    elif scenario == "load_transients_train":
        block = max(sample_count // 14, 1)
        for channel in range(4):
            levels = rng.uniform(-0.85 if channel == 2 else 0.02, 1.05, size=15)
            states[:, channel] = levels[np.minimum(np.arange(sample_count) // block, 14)]
    elif scenario in {
        "high_maneuver_train",
        "rapid_reversal_train",
        "correction_high_dynamic",
        "holdout_high_maneuver",
        "preflight_anchor_train_a",
        "preflight_anchor_train_b",
        "preflight_anchor_holdout",
    }:
        # High-dynamic calibration families deliberately cover larger command
        # amplitudes and faster, simultaneous transitions than normal cruise.
        # The holdout uses different event timing/frequency so it remains an
        # independent interpolation test rather than a waveform replay.
        p = t / max(t[-1], 1.0)

        def pulse(start: float, stop: float, edge: float = 0.006) -> np.ndarray:
            return 0.5 * (
                np.tanh((p - start) / edge) - np.tanh((p - stop) / edge)
            )

        if scenario == "high_maneuver_train":
            events = pulse(0.10, 0.18) + 0.82 * pulse(0.34, 0.43) + 1.08 * pulse(0.70, 0.79)
            reversal = pulse(0.51, 0.59)
            states[:, 0] = 0.26 + 1.32 * events - 1.18 * reversal + 0.18 * np.sin(2 * np.pi * 5.7 * p)
            states[:, 1] = 0.22 + 0.73 * np.abs(states[:, 0]) + 0.42 * pulse(0.28, 0.48)
            states[:, 2] = 1.24 * np.sin(2 * np.pi * 4.3 * p) * (0.35 + 0.65 * events)
            states[:, 3] = 0.38 + 0.43 * np.abs(states[:, 0]) + 0.35 * pulse(0.66, 0.86)
        elif scenario == "rapid_reversal_train":
            carrier = np.sin(2 * np.pi * 6.2 * p)
            envelope = 0.35 + 1.05 * (pulse(0.16, 0.31) + pulse(0.56, 0.75))
            states[:, 0] = envelope * carrier
            states[:, 1] = 0.30 + 0.82 * np.abs(states[:, 0])
            states[:, 2] = 1.35 * np.sign(np.sin(2 * np.pi * 4.7 * p + 0.3)) * pulse(0.25, 0.88)
            states[:, 3] = 0.35 + 0.32 * np.abs(states[:, 2]) + 0.24 * pulse(0.48, 0.67)
        elif scenario == "correction_high_dynamic":
            events = pulse(0.13, 0.22) + pulse(0.46, 0.57) + 0.9 * pulse(0.77, 0.88)
            states[:, 0] = 0.18 + 1.42 * events - 1.08 * pulse(0.31, 0.38)
            states[:, 1] = 0.24 + 0.78 * np.abs(states[:, 0]) + 0.18 * np.sin(2 * np.pi * 3.8 * p)
            states[:, 2] = 1.32 * np.sin(2 * np.pi * 5.1 * p + 0.8) * (0.35 + 0.65 * events)
            states[:, 3] = 0.42 + 0.38 * events + 0.16 * np.sin(2 * np.pi * 7.2 * p)
        elif scenario == "holdout_high_maneuver":
            events = pulse(0.08, 0.15) + 0.92 * pulse(0.39, 0.49) + 1.10 * pulse(0.82, 0.90)
            states[:, 0] = 0.20 + 1.38 * events - 1.24 * pulse(0.62, 0.70) + 0.13 * np.sin(2 * np.pi * 6.9 * p)
            states[:, 1] = 0.27 + 0.76 * np.abs(states[:, 0]) + 0.31 * pulse(0.52, 0.73)
            states[:, 2] = 1.28 * np.sin(2 * np.pi * 5.6 * p + 1.1) * (0.30 + 0.70 * events)
            states[:, 3] = 0.40 + 0.39 * np.abs(states[:, 0]) + 0.29 * pulse(0.74, 0.94)
        elif scenario == "preflight_anchor_train_a":
            # Sequential and overlapping commissioning pulses excite all four
            # source channels without replaying a navigation trajectory.
            states[:, 0] = 0.14 + 1.48 * pulse(0.06, 0.19) - 1.20 * pulse(0.57, 0.68)
            states[:, 1] = 0.16 + 1.34 * pulse(0.22, 0.38) + 0.42 * pulse(0.72, 0.90)
            states[:, 2] = 1.38 * pulse(0.39, 0.54) - 1.28 * pulse(0.76, 0.88)
            states[:, 3] = 0.20 + 1.30 * pulse(0.14, 0.31) + 0.37 * pulse(0.49, 0.73)
        elif scenario == "preflight_anchor_train_b":
            gate = 0.35 + 0.95 * (pulse(0.10, 0.27) + pulse(0.52, 0.81))
            states[:, 0] = 0.16 + gate * np.sin(2 * np.pi * 5.3 * p + 0.2)
            states[:, 1] = 0.20 + 0.86 * np.abs(states[:, 0]) + 0.24 * np.sin(2 * np.pi * 3.7 * p)
            states[:, 2] = 1.32 * np.sin(2 * np.pi * 4.4 * p + 0.9) * gate
            states[:, 3] = 0.28 + 0.38 * np.abs(states[:, 2]) + 0.31 * pulse(0.33, 0.64)
        else:
            gate = 0.32 + 0.98 * (pulse(0.18, 0.36) + pulse(0.66, 0.87))
            states[:, 0] = 0.19 + 1.18 * gate * np.sin(2 * np.pi * 5.9 * p + 0.6)
            states[:, 1] = 0.25 + 0.79 * np.abs(states[:, 0]) + 0.29 * pulse(0.42, 0.61)
            states[:, 2] = 1.25 * np.sin(2 * np.pi * 4.9 * p + 1.3) * gate
            states[:, 3] = 0.34 + 0.34 * np.abs(states[:, 2]) + 0.27 * pulse(0.08, 0.24)
    elif scenario == "correction_mixed":
        for channel in range(4):
            sweep(channel, -0.15 if channel == 2 else 0.10, 0.95, 1.0 + 0.35 * channel)
    elif scenario == "holdout_converter_actuator":
        sweep(1, 0.10, 1.0, 2.3)
        states[:, 2] = 0.80 * np.sin(2.0 * np.pi * 2.1 * t / max(t[-1], 1.0) + 1.0)
        sweep(3, 0.25, 0.85, 0.7)
    elif scenario == "holdout_all_sources":
        for channel in range(4):
            sweep(channel, -0.25 if channel == 2 else 0.08, 1.0, 0.8 + 0.47 * channel)
        states[:, 0] *= 0.75 + 0.20 * (states[:, 1] > 0.5)
    else:
        raise ValueError("Unknown source scenario: %s" % scenario)

    high_dynamic = scenario in {
        "high_maneuver_train",
        "rapid_reversal_train",
        "correction_high_dynamic",
        "holdout_high_maneuver",
        "preflight_anchor_train_a",
        "preflight_anchor_train_b",
        "preflight_anchor_holdout",
    }
    if high_dynamic:
        states[:, 0] = np.clip(states[:, 0], -1.55, 1.75)
        states[:, [1, 3]] = np.clip(states[:, [1, 3]], 0.0, 1.70)
        states[:, 2] = np.clip(states[:, 2], -1.45, 1.45)
    else:
        states[:, [0, 1, 3]] = np.clip(states[:, [0, 1, 3]], 0.0, 1.1)
        states[:, 2] = np.clip(states[:, 2], -1.0, 1.0)
    temperature = slow_temperature + 0.25 * states[:, 0] + 0.12 * states[:, 1]
    return states, temperature
