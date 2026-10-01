from typing import Dict

import numpy as np

from .models import CoilResponseModel


class FeedforwardController:
    """Frozen-model feedforward controller; it never receives cesium data."""

    def __init__(
        self,
        coil_model: CoilResponseModel,
        regularization: float,
        gradient_weight: float,
        current_limit_a: float,
        current_slew_a_per_s: float,
        dt_s: float,
        driver_tau_s: float = 0.0,
        driver_preemphasis_enabled: bool = False,
    ):
        self.coil_model = coil_model
        self.regularization = float(regularization)
        self.gradient_weight = float(gradient_weight)
        self.current_limit_a = float(current_limit_a)
        self.max_step = float(current_slew_a_per_s) * float(dt_s)
        self.dt_s = float(dt_s)
        self.driver_tau_s = float(driver_tau_s)
        self.driver_preemphasis_enabled = bool(driver_preemphasis_enabled)

    def _solve_sample(
        self, predicted_source: np.ndarray, temperature_c: float, mode: str
    ) -> np.ndarray:
        matrix = self.coil_model.matrix_at_temperature(temperature_c)
        channel_count = matrix.shape[1]
        if mode == "none":
            return np.zeros(channel_count)
        bias = self.coil_model.intercept
        if mode == "uniform":
            outputs = [0, 1, 2]
            channels = [0, 1, 2]
            weights = np.ones(3)
        elif mode == "joint_main":
            outputs = [0, 1, 2, 3]
            channels = [0, 1, 2, 3]
            weights = np.array([1.0, 1.0, 1.0, self.gradient_weight])
        elif mode == "joint":
            # A source-free symmetric gradient tensor has five independent
            # components, so high-gradient protection uses five identified
            # gradient channels in addition to the three uniform-field axes.
            outputs = list(range(8))
            channels = list(range(8))
            weights = np.array([1.0, 1.0, 1.0] + [self.gradient_weight] * 5)
        else:
            raise ValueError("Unknown compensation mode: %s" % mode)
        response = matrix[np.ix_(outputs, channels)]
        target = -(predicted_source[outputs] + bias[outputs])
        weighted_response = response * weights[:, None]
        weighted_target = target * weights
        lhs = weighted_response.T.dot(weighted_response)
        lhs += self.regularization * np.eye(len(channels))
        solution = np.linalg.solve(lhs, weighted_response.T.dot(weighted_target))
        command = np.zeros(channel_count)
        command[channels] = solution
        return np.clip(command, -self.current_limit_a, self.current_limit_a)

    def commands(
        self, predicted_source_state8: np.ndarray, temperature_c: np.ndarray, mode: str
    ) -> np.ndarray:
        channel_count = self.coil_model.matrix_at_temperature(float(temperature_c[0])).shape[1]
        desired = np.zeros((len(predicted_source_state8), channel_count), dtype=float)
        for index in range(len(desired)):
            desired[index] = self._solve_sample(
                predicted_source_state8[index], temperature_c[index], mode
            )
        # Causal inverse of the identified first-order current driver.  With
        # actual[k] = actual[k-1] + alpha*(command[k]-actual[k-1]), this
        # command makes actual[k] track desired[k] exactly when unsaturated.
        # It uses only the current and previous feedforward demand.
        if (
            self.driver_preemphasis_enabled
            and self.driver_tau_s > 0.0
            and len(desired) > 1
            and mode != "none"
        ):
            alpha = 1.0 - np.exp(-self.dt_s / max(self.driver_tau_s, 1.0e-4))
            emphasized = desired.copy()
            emphasized[1:] = desired[:-1] + (desired[1:] - desired[:-1]) / alpha
            desired = np.clip(emphasized, -self.current_limit_a, self.current_limit_a)
        if np.isinf(self.max_step):
            return desired
        # Slew limiting is causal and belongs to the controller command path.
        limited = np.zeros_like(desired)
        # The navigation run starts after the feedforward controller has been
        # initialized from the measured pre-dive state; do not manufacture a
        # one-sample zero-current transient at t=0.
        limited[0] = np.clip(desired[0], -self.current_limit_a, self.current_limit_a)
        for index in range(1, len(limited)):
            delta = np.clip(desired[index] - limited[index - 1], -self.max_step, self.max_step)
            limited[index] = limited[index - 1] + delta
        return limited


def command_metrics(command_a: np.ndarray, current_limit_a: float) -> Dict[str, float]:
    absolute = np.abs(command_a)
    return {
        "peak_current_a": float(absolute.max()),
        "rms_current_a": float(np.sqrt(np.mean(command_a ** 2))),
        "saturation_fraction": float(np.mean(absolute >= 0.999 * current_limit_a)),
    }
