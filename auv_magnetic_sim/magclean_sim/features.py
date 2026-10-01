from typing import List, Tuple

import numpy as np


def _delayed(matrix: np.ndarray, delay: int) -> np.ndarray:
    if delay <= 0:
        return matrix.copy()
    prefix = np.repeat(matrix[:1], delay, axis=0)
    return np.vstack([prefix, matrix[:-delay]])


def _first_order_bank(matrix: np.ndarray, sample_rate_hz: float, time_constants) -> List[np.ndarray]:
    """Return causal first-order state estimates for a bank of time constants."""
    dt_s = 1.0 / float(sample_rate_hz)
    outputs = []
    for tau_s in time_constants:
        filtered = np.zeros_like(matrix)
        filtered[0] = matrix[0]
        alpha = 1.0 - np.exp(-dt_s / max(float(tau_s), 1.0e-6))
        for sample in range(1, len(matrix)):
            filtered[sample] = filtered[sample - 1] + alpha * (
                matrix[sample] - filtered[sample - 1]
            )
        outputs.append(filtered)
    return outputs


def build_source_features(
    states: np.ndarray,
    temperature_c: np.ndarray,
    history: int = 2,
    sample_rate_hz: float = 20.0,
    variant: str = "enhanced_dynamic",
) -> Tuple[np.ndarray, List[str]]:
    states = np.asarray(states, dtype=float)
    temperature = np.asarray(temperature_c, dtype=float).reshape(-1, 1)
    if states.shape[0] != temperature.shape[0]:
        raise ValueError("State and temperature lengths differ")
    features = [states]
    names = ["s%d" % (index + 1) for index in range(states.shape[1])]
    if variant == "legacy":
        maximum_history = history
    elif variant == "enhanced_dynamic":
        # Cover 0.4 s of causal history at the configured sample rate. The
        # bank below then represents longer memory without a huge delay vector.
        maximum_history = max(history, int(round(0.4 * sample_rate_hz)))
    else:
        raise ValueError("Unknown source model variant: %s" % variant)
    for delay in range(1, maximum_history + 1):
        features.append(_delayed(states, delay))
        names.extend(["s%d_d%d" % (index + 1, delay) for index in range(states.shape[1])])
    features.append(states ** 2)
    names.extend(["s%d_sq" % (index + 1) for index in range(states.shape[1])])
    cross = []
    for left in range(states.shape[1]):
        for right in range(left + 1, states.shape[1]):
            cross.append((states[:, left] * states[:, right])[:, None])
            names.append("s%dxs%d" % (left + 1, right + 1))
    if cross:
        features.append(np.hstack(cross))
    normalized_temperature = (temperature - 20.0) / 10.0
    features.append(states * normalized_temperature)
    names.extend(["s%d_temp" % (index + 1) for index in range(states.shape[1])])
    if variant == "enhanced_dynamic":
        # These terms can represent delayed electromagnetic/thermal source
        # response, nonlinear source strength, and temperature-dependent
        # interactions while remaining a stable convex regression problem.
        delay_one = _delayed(states, 1)
        features.append((states - delay_one) * sample_rate_hz)
        names.extend(["ds%d_dt" % (index + 1) for index in range(states.shape[1])])

        features.append(states ** 2 * normalized_temperature)
        names.extend(["s%d_sq_temp" % (index + 1) for index in range(states.shape[1])])
        if cross:
            cross_matrix = np.hstack(cross)
            features.append(cross_matrix * normalized_temperature)
            names.extend(
                ["cross%d_temp" % (index + 1) for index in range(cross_matrix.shape[1])]
            )

        time_constants = (0.04, 0.06, 0.08, 0.12, 0.16, 0.22, 0.32, 0.48)
        for tau_s, filtered in zip(
            time_constants, _first_order_bank(states, sample_rate_hz, time_constants)
        ):
            suffix = int(round(1000.0 * tau_s))
            features.append(filtered)
            names.extend(["s%d_ema%03d" % (index + 1, suffix) for index in range(states.shape[1])])
            features.append(filtered * normalized_temperature)
            names.extend(
                ["s%d_ema%03d_temp" % (index + 1, suffix) for index in range(states.shape[1])]
            )
    return np.hstack(features), names


def build_source_alignment_features(
    states: np.ndarray, temperature_c: np.ndarray, sample_rate_hz: float
) -> Tuple[np.ndarray, List[str]]:
    """Compact physics-guided features for short preflight drift alignment.

    Unlike the high-capacity base estimator, this adapter contains only the
    terms needed to describe source gain changes: causal source memory,
    quadratic response, neighbour coupling and their temperature dependence.
    This reduces the chance of fitting a commissioning waveform itself.
    """
    states = np.asarray(states, dtype=float)
    temperature = np.asarray(temperature_c, dtype=float).reshape(-1, 1)
    normalized_temperature = (temperature - 20.0) / 10.0
    features = []
    names = []
    time_constants = (0.04, 0.06, 0.08, 0.12, 0.16, 0.22, 0.32, 0.48)
    for tau_s, filtered in zip(
        time_constants, _first_order_bank(states, sample_rate_hz, time_constants)
    ):
        suffix = int(round(1000.0 * tau_s))
        features.extend([filtered, filtered * normalized_temperature])
        names.extend(["align_s%d_ema%03d" % (i + 1, suffix) for i in range(4)])
        names.extend(
            ["align_s%d_ema%03d_temp" % (i + 1, suffix) for i in range(4)]
        )
    squared = states ** 2
    directed_cross = np.column_stack(
        [states[:, index] * states[:, (index + 1) % 4] for index in range(4)]
    )
    features.extend(
        [
            squared,
            directed_cross,
            squared * normalized_temperature,
            directed_cross * normalized_temperature,
        ]
    )
    names.extend(["align_s%d_sq" % (i + 1) for i in range(4)])
    names.extend(["align_s%dxnext" % (i + 1) for i in range(4)])
    names.extend(["align_s%d_sq_temp" % (i + 1) for i in range(4)])
    names.extend(["align_s%dxnext_temp" % (i + 1) for i in range(4)])
    return np.hstack(features), names


def build_coil_features(
    currents_a: np.ndarray, temperature_c: np.ndarray
) -> Tuple[np.ndarray, List[str]]:
    currents = np.asarray(currents_a, dtype=float)
    temperature = np.asarray(temperature_c, dtype=float).reshape(-1, 1)
    delta_temperature = (temperature - 20.0) / 10.0
    features = np.hstack([currents, currents * delta_temperature])
    names = ["i%d" % (index + 1) for index in range(currents.shape[1])]
    names += ["i%d_temp" % (index + 1) for index in range(currents.shape[1])]
    return features, names
