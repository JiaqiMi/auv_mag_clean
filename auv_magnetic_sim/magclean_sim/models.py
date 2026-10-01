from typing import Dict, Optional

import numpy as np

from .features import build_coil_features, build_source_features


class RidgeMultiOutput:
    """Small dependency-free multi-output ridge regressor.

    Inputs are standardized with statistics fitted on the calibration split.
    The output is kept in engineering units so every prediction remains easy
    to audit.
    """

    def __init__(self, ridge: float):
        self.ridge = float(ridge)
        self.x_mean = None
        self.x_scale = None
        self.y_mean = None
        self.coefficient = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeMultiOutput":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.x_mean = x.mean(axis=0)
        self.x_scale = x.std(axis=0)
        self.x_scale[self.x_scale < 1.0e-9] = 1.0
        self.y_mean = y.mean(axis=0)
        xs = (x - self.x_mean) / self.x_scale
        lhs = xs.T.dot(xs) + self.ridge * np.eye(xs.shape[1])
        self.coefficient = np.linalg.solve(lhs, xs.T.dot(y - self.y_mean))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coefficient is None:
            raise RuntimeError("Model has not been fitted")
        xs = (np.asarray(x, dtype=float) - self.x_mean) / self.x_scale
        return self.y_mean + xs.dot(self.coefficient)


class SourceFieldModel:
    """Frozen source-to-field model plus a bounded offline residual model."""

    def __init__(
        self,
        ridge: float,
        residual_limit: float,
        sample_rate_hz: float = 20.0,
        variant: str = "enhanced_dynamic",
    ):
        self.base = RidgeMultiOutput(ridge)
        self.correction = RidgeMultiOutput(ridge * 2.0)
        self.residual_limit = float(residual_limit)
        self.sample_rate_hz = float(sample_rate_hz)
        self.variant = variant
        self.correction_scale = 0.0
        self.correction_ridge_selected = None
        self.correction_validation_improvement = 0.0
        self.feature_names = []

    def features(self, states: np.ndarray, temperature_c: np.ndarray) -> np.ndarray:
        features, names = build_source_features(
            states,
            temperature_c,
            sample_rate_hz=self.sample_rate_hz,
            variant=self.variant,
        )
        self.feature_names = names
        return features

    def fit_base(self, features: np.ndarray, measured_state8: np.ndarray) -> None:
        self.base.fit(features, measured_state8)

    def fit_correction(self, features: np.ndarray, measured_state8: np.ndarray) -> None:
        base_prediction = self.base.predict(features)
        residual = measured_state8 - base_prediction
        split = max(int(round(0.70 * len(features))), 1)
        split = min(split, len(features) - 1)
        train_x, validation_x = features[:split], features[split:]
        train_residual, validation_residual = residual[:split], residual[split:]
        component_scale = np.std(measured_state8[split:], axis=0)
        component_scale = np.maximum(component_scale, np.array([1, 1, 1, 10, 10, 10, 10, 10]))
        baseline_score = float(np.mean((validation_residual / component_scale) ** 2))
        best_score = baseline_score
        best_ridge = None
        candidates = sorted(
            set(
                [
                    max(self.base.ridge * 2.0, 1.0e-9),
                    1.0e-5,
                    1.0e-4,
                    1.0e-3,
                    1.0e-2,
                    1.0e-1,
                    1.0,
                ]
            )
        )
        for ridge in candidates:
            candidate = RidgeMultiOutput(ridge).fit(train_x, train_residual)
            candidate_error = validation_residual - candidate.predict(validation_x)
            score = float(np.mean((candidate_error / component_scale) ** 2))
            if score < best_score:
                best_score = score
                best_ridge = ridge
        # Correction must demonstrate a genuine improvement on a part of the
        # correction scenario that was not used to fit it.
        if best_ridge is None or best_score > 0.98 * baseline_score:
            self.correction_scale = 0.0
            self.correction_ridge_selected = None
            self.correction_validation_improvement = 0.0
            return
        self.correction = RidgeMultiOutput(best_ridge).fit(features, residual)
        self.correction_ridge_selected = best_ridge
        self.correction_validation_improvement = 1.0 - best_score / max(
            baseline_score, 1.0e-15
        )
        raw_correction = self.correction.predict(features)
        base_rms = float(np.sqrt(np.mean(base_prediction ** 2)))
        correction_rms = float(np.sqrt(np.mean(raw_correction ** 2)))
        allowed = self.residual_limit * max(base_rms, 1.0e-9)
        self.correction_scale = min(1.0, allowed / max(correction_rms, 1.0e-9))

    def predict(self, features: np.ndarray, use_correction: bool = True) -> np.ndarray:
        base = self.base.predict(features)
        if not use_correction or self.correction_scale <= 0.0:
            return base
        return base + self.correction_scale * self.correction.predict(features)

    def predict_from_states(
        self, states: np.ndarray, temperature_c: np.ndarray, use_correction: bool = True
    ) -> np.ndarray:
        return self.predict(self.features(states, temperature_c), use_correction)

    def summary(self) -> Dict[str, float]:
        return {
            "ridge": self.base.ridge,
            "variant": self.variant,
            "feature_count": len(self.feature_names),
            "residual_limit": self.residual_limit,
            "correction_scale": self.correction_scale,
            "correction_ridge_selected": self.correction_ridge_selected,
            "correction_validation_improvement": self.correction_validation_improvement,
        }


class PreflightSourceAlignment:
    """Frozen residual adapter fitted only on preflight commissioning data.

    The base estimator remains unchanged.  This model estimates the small
    transfer residual caused by source-gain drift from the same source-state
    features, then is frozen before the navigation evaluation begins.
    """

    def __init__(self, ridge: float):
        self.model = RidgeMultiOutput(ridge)
        self.fitted = False

    def fit(
        self,
        source_features: np.ndarray,
        base_prediction: np.ndarray,
        observed_state8: np.ndarray,
    ) -> None:
        inputs = np.column_stack([source_features, base_prediction])
        self.model.fit(inputs, observed_state8 - base_prediction)
        self.fitted = True

    def predict(
        self, source_features: np.ndarray, base_prediction: np.ndarray
    ) -> np.ndarray:
        if not self.fitted:
            return np.asarray(base_prediction, dtype=float)
        inputs = np.column_stack([source_features, base_prediction])
        return np.asarray(base_prediction, dtype=float) + self.model.predict(inputs)

    def summary(self) -> Dict[str, object]:
        return {
            "enabled": self.fitted,
            "ridge": self.model.ridge,
            "input_feature_count": 0
            if self.model.coefficient is None
            else int(self.model.coefficient.shape[0]),
        }


class CoilResponseModel:
    """Identified linear temperature-dependent coil response.

    The explicit coefficient form is retained because the controller needs an
    auditable current-to-[field, gradient] matrix at each temperature.
    """

    def __init__(self, ridge: float):
        self.ridge = float(ridge)
        self.intercept = np.zeros(8)
        self.coefficient = None
        self.feature_names = []

    def fit(
        self, currents_a: np.ndarray, temperature_c: np.ndarray, measured_state8: np.ndarray
    ) -> None:
        features, names = build_coil_features(currents_a, temperature_c)
        design = np.column_stack([np.ones(len(features)), features])
        penalty = self.ridge * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T.dot(design) + penalty, design.T.dot(measured_state8)
        )
        self.intercept = coefficients[0]
        self.coefficient = coefficients[1:]
        self.feature_names = names

    def predict(self, currents_a: np.ndarray, temperature_c: np.ndarray) -> np.ndarray:
        if self.coefficient is None:
            raise RuntimeError("Coil response model has not been fitted")
        features, _ = build_coil_features(currents_a, temperature_c)
        return self.intercept + features.dot(self.coefficient)

    def matrix_at_temperature(self, temperature_c: float) -> np.ndarray:
        if self.coefficient is None:
            raise RuntimeError("Coil response model has not been fitted")
        delta = (float(temperature_c) - 20.0) / 10.0
        # coefficient maps row-vector features to row-vector state output.
        channel_count = self.coefficient.shape[0] // 2
        return (
            self.coefficient[:channel_count]
            + delta * self.coefficient[channel_count : 2 * channel_count]
        ).T

    def summary(self) -> Dict[str, object]:
        return {
            "ridge": self.ridge,
            "intercept_state8": self.intercept.tolist(),
            "matrix_at_20c": self.matrix_at_temperature(20.0).tolist(),
        }


def component_metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, object]:
    error = np.asarray(predicted) - np.asarray(observed)
    component_rmse = np.sqrt(np.mean(error ** 2, axis=0))
    return {
        "field_rmse_nt": float(np.sqrt(np.mean(error[:, :3] ** 2))),
        "gradient_rmse_nt_per_m": float(np.sqrt(np.mean(error[:, 3:] ** 2))),
        "component_rmse": component_rmse.tolist(),
    }
