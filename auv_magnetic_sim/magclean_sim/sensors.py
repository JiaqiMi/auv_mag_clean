from typing import Tuple

import numpy as np

from .math3d import (
    gradient_stencil_points,
    reconstruct_state8,
    state8_to_array_field,
    stencil_fields_to_state8,
)


# Removable bench fixture: one reference point at the future cesium sensitive
# volume plus symmetric +/- axis points. The 10 mm half-baseline keeps the
# central-difference truncation error small for nearby onboard dipoles.
DEFAULT_ARRAY_BASELINE_M = 0.01
DEFAULT_ARRAY_POSITIONS_M = gradient_stencil_points(DEFAULT_ARRAY_BASELINE_M)


class BenchVectorArray:
    def __init__(
        self,
        positions_m: np.ndarray = None,
        noise_nt: float = 0.18,
        rng: np.random.RandomState = None,
        position_error_mm: float = 0.0,
        axis_error_deg: float = 0.0,
        gain_error_fraction: float = 0.0,
    ):
        self.positions_m = np.asarray(
            DEFAULT_ARRAY_POSITIONS_M if positions_m is None else positions_m, dtype=float
        )
        self.noise_nt = float(noise_nt)
        calibration_rng = rng or np.random.RandomState(0)
        self.true_positions_m = self.positions_m + calibration_rng.normal(
            0.0, float(position_error_mm) * 1.0e-3, size=self.positions_m.shape
        )
        self.gain = 1.0 + calibration_rng.normal(
            0.0, float(gain_error_fraction), size=(len(self.positions_m), 3)
        )
        angle_sigma = np.deg2rad(float(axis_error_deg))
        self.axis_matrices = np.zeros((len(self.positions_m), 3, 3), dtype=float)
        for index in range(len(self.positions_m)):
            rotation = calibration_rng.normal(0.0, angle_sigma, size=3)
            skew = np.array(
                [
                    [0.0, -rotation[2], rotation[1]],
                    [rotation[2], 0.0, -rotation[0]],
                    [-rotation[1], rotation[0], 0.0],
                ]
            )
            self.axis_matrices[index] = np.eye(3) + skew

    def _apply_sensor_errors(self, ideal_fields_nt: np.ndarray) -> np.ndarray:
        fields = np.asarray(ideal_fields_nt, dtype=float)
        if fields.ndim == 2:
            rotated = np.einsum("pij,pj->pi", self.axis_matrices, fields)
        else:
            rotated = np.einsum("pij,npj->npi", self.axis_matrices, fields)
        return rotated * self.gain

    def _reconstruct(self, measured: np.ndarray) -> np.ndarray:
        if (
            self.positions_m.shape == DEFAULT_ARRAY_POSITIONS_M.shape
            and np.allclose(self.positions_m, DEFAULT_ARRAY_POSITIONS_M)
        ):
            return stencil_fields_to_state8(measured, DEFAULT_ARRAY_BASELINE_M)
        return reconstruct_state8(measured, self.positions_m)

    def observe_state8(
        self, state8: np.ndarray, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray]:
        ideal = state8_to_array_field(state8, self.true_positions_m)
        measured_ideal = self._apply_sensor_errors(ideal)
        measured = measured_ideal + rng.normal(0.0, self.noise_nt, size=ideal.shape)
        reconstructed = self._reconstruct(measured)
        return measured, reconstructed

    def observe_fields(
        self, ideal_fields_nt: np.ndarray, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray]:
        measured_ideal = self._apply_sensor_errors(ideal_fields_nt)
        measured = measured_ideal + rng.normal(
            0.0, self.noise_nt, size=ideal_fields_nt.shape
        )
        reconstructed = self._reconstruct(measured)
        return measured, reconstructed


def apply_cesium_scalar_sensor(
    ideal_total_field_nt: np.ndarray,
    dt_s: float,
    bandwidth_hz: float,
    noise_nt: float,
    white_noise: np.ndarray,
    random_walk: np.ndarray,
) -> np.ndarray:
    ideal = np.asarray(ideal_total_field_nt, dtype=float)
    alpha = 1.0 - np.exp(-2.0 * np.pi * max(bandwidth_hz, 0.05) * dt_s)
    filtered = np.empty_like(ideal)
    filtered[0] = ideal[0]
    for index in range(1, len(ideal)):
        filtered[index] = filtered[index - 1] + alpha * (ideal[index] - filtered[index - 1])
    return filtered + noise_nt * white_noise + 0.004 * random_walk
