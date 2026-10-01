from typing import Tuple

import numpy as np

from .math3d import state8_to_array_field


class CoilSystem:
    """Physical 3-uniform + 5-gradient channel response used by the forward model."""

    def __init__(
        self,
        drift_fraction: float = 0.0,
        uniform_gain_scale: float = 1.0,
        gradient_gain_scale: float = 1.0,
    ):
        self.base_matrix = np.array(
            [
                [122.0, 3.8, -2.2, 1.4, -1.0, 0.8, 0.6, -0.5],
                [3.2, 116.0, 2.8, -1.8, 1.2, -0.9, 0.7, 0.5],
                [-2.7, 4.1, 126.0, 0.9, -0.7, 1.0, -0.6, 0.8],
                [1.8, 0.8, -0.5, 82.0, -5.0, 4.0, 2.0, -2.0],
                [-0.7, 1.1, 0.5, -37.0, 78.0, -4.0, 3.0, 2.0],
                [0.9, 1.2, 0.3, 4.5, -3.0, 76.0, 2.0, -2.0],
                [0.4, -0.6, 1.0, 3.2, 2.0, -3.0, 80.0, 3.0],
                [0.7, 0.4, -0.8, -2.8, 2.0, 3.0, -2.0, 79.0],
            ],
            dtype=float,
        )
        self.base_matrix[:, :3] *= float(uniform_gain_scale)
        self.base_matrix[:, 3:] *= float(gradient_gain_scale)
        column_pattern = np.array([0.7, -0.5, 1.0, -0.8, 0.6, -0.4, 0.9, -0.7])
        self.base_matrix *= 1.0 + float(drift_fraction) * column_pattern[None, :]
        self.temperature_coefficients = np.array(
            [-0.0017, -0.0013, -0.0018, -0.0022, -0.0020, -0.0021, -0.0019, -0.0023]
        )
        self.bias_state = np.array([0.05, -0.03, 0.04, 0.02, -0.01, 0.0, 0.0, 0.0])

    def matrix_at_temperature(self, temperature_c: float) -> np.ndarray:
        scale = 1.0 + self.temperature_coefficients * (float(temperature_c) - 20.0)
        return self.base_matrix * scale[None, :]

    def response_state(self, currents_a: np.ndarray, temperature_c: np.ndarray) -> np.ndarray:
        currents = np.asarray(currents_a, dtype=float)
        temperature = np.asarray(temperature_c, dtype=float)
        response = np.zeros((len(currents), 8), dtype=float)
        for index in range(len(currents)):
            matrix = self.matrix_at_temperature(temperature[index])
            linear = matrix.dot(currents[index])
            nonlinear = 0.45 * np.tanh(currents[index]).sum() * np.array(
                [0.03, -0.02, 0.01, 0.05, -0.02, 0.01, 0.0, 0.0]
            )
            response[index] = linear + nonlinear + self.bias_state
        return response

    def array_field(
        self, currents_a: np.ndarray, temperature_c: np.ndarray, positions_m: np.ndarray
    ) -> np.ndarray:
        return state8_to_array_field(self.response_state(currents_a, temperature_c), positions_m)


def generate_coil_calibration_currents(
    sample_count_per_channel: int,
    sample_rate_hz: float,
    rng: np.random.RandomState,
    current_limit_a: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train currents/temperature and scenario labels."""
    blocks = []
    temperatures = []
    labels = []
    n = sample_count_per_channel
    t = np.arange(n, dtype=float) / sample_rate_hz
    channel_count = 8
    for channel in range(channel_count):
        current = np.zeros((n, channel_count), dtype=float)
        wave = np.sin(2.0 * np.pi * (1.0 + 0.25 * channel) * t / max(t[-1], 1.0))
        steps = ((np.arange(n) // max(n // 8, 1)) % 8 - 3.5) / 3.5
        current[:, channel] = current_limit_a * (0.55 * wave + 0.35 * steps)
        blocks.append(current)
        temperatures.append(20.0 + 4.0 * np.linspace(0.0, 1.0, n) + 0.2 * channel)
        labels.extend(["single_channel_%d" % (channel + 1)] * n)
    combined = rng.uniform(
        -0.75 * current_limit_a,
        0.75 * current_limit_a,
        size=(n, channel_count),
    )
    # Smooth random combinations to create a validation trajectory.
    for index in range(1, n):
        combined[index] = 0.82 * combined[index - 1] + 0.18 * combined[index]
    blocks.append(combined)
    temperatures.append(23.0 + 2.0 * np.sin(2.0 * np.pi * t / max(t[-1], 1.0)))
    labels.extend(["holdout_multichannel"] * n)
    return np.vstack(blocks), np.concatenate(temperatures), np.asarray(labels)


def simulate_driver(
    command_a: np.ndarray,
    dt_s: float,
    tau_s: float,
    current_limit_a: float,
) -> np.ndarray:
    command = np.asarray(command_a, dtype=float)
    actual = np.zeros_like(command)
    alpha = 1.0 - np.exp(-dt_s / max(tau_s, 1.0e-4))
    actual[0] = np.clip(command[0], -current_limit_a, current_limit_a)
    for index in range(1, len(command)):
        actual[index] = actual[index - 1] + alpha * (command[index] - actual[index - 1])
        actual[index] = np.clip(actual[index], -current_limit_a, current_limit_a)
    return actual
