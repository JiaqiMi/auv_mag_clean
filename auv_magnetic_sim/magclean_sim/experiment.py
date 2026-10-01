import csv
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .calibration import CalibrationBundle, run_bench_calibration
from .coils import CoilSystem, simulate_driver
from .config import COIL_CHANNEL_LABELS, SOURCE_STATE_LABELS, STATE_VECTOR_LABELS, SimulationConfig
from .controller import FeedforwardController, command_metrics
from .environment import generate_navigation_profile
from .features import build_source_alignment_features
from .sensors import apply_cesium_scalar_sensor
from .sources import MagneticSourceSystem


def _downsample_indices(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=int)
    return np.unique(np.linspace(0, length - 1, maximum).astype(int))


def _line_payload(x: np.ndarray, series: Dict[str, np.ndarray], maximum: int) -> Dict[str, object]:
    index = _downsample_indices(len(x), maximum)
    return {
        "x": np.asarray(x)[index].round(7).tolist(),
        "series": {
            name: np.asarray(values)[index].round(7).tolist() for name, values in series.items()
        },
    }


def _principal_gradient_magnitude(state8: np.ndarray) -> np.ndarray:
    """Largest absolute principal value of the symmetric traceless gradient tensor."""
    state = np.asarray(state8, dtype=float)
    tensor = np.zeros((len(state), 3, 3), dtype=float)
    tensor[:, 0, 0] = state[:, 3]
    tensor[:, 1, 1] = state[:, 4]
    tensor[:, 2, 2] = -(state[:, 3] + state[:, 4])
    tensor[:, 0, 1] = tensor[:, 1, 0] = state[:, 5]
    tensor[:, 0, 2] = tensor[:, 2, 0] = state[:, 6]
    tensor[:, 1, 2] = tensor[:, 2, 1] = state[:, 7]
    return np.max(np.abs(np.linalg.eigvalsh(tensor)), axis=1)


def _count_true_runs(mask: np.ndarray) -> int:
    active = np.asarray(mask, dtype=bool).astype(int)
    if not len(active):
        return 0
    return int(active[0] + np.sum(np.diff(active) == 1))


def _gradient_compliance(
    state8: np.ndarray,
    sample_rate_hz: float,
    warning_nt_per_m: float,
    tolerance_nt_per_m: float,
) -> Dict[str, float]:
    magnitude = _principal_gradient_magnitude(state8)
    warning = magnitude > warning_nt_per_m
    exceeded = magnitude > tolerance_nt_per_m
    return {
        "principal_peak_nt_per_m": float(np.max(magnitude)),
        "principal_rms_nt_per_m": float(np.sqrt(np.mean(magnitude ** 2))),
        "warning_exceed_fraction": float(np.mean(warning)),
        "warning_exceed_duration_s": float(np.sum(warning) / sample_rate_hz),
        "tolerance_exceed_fraction": float(np.mean(exceeded)),
        "tolerance_exceed_duration_s": float(np.sum(exceeded) / sample_rate_hz),
        "tolerance_exceed_event_count": _count_true_runs(exceeded),
        "valid_fraction_by_gradient": float(np.mean(~exceeded)),
    }


def _residual_metrics(
    source_state: np.ndarray,
    coil_state: np.ndarray,
    command_a: np.ndarray,
    config: SimulationConfig,
) -> Dict[str, float]:
    residual = source_state + coil_state
    field_before = np.linalg.norm(source_state[:, :3], axis=1)
    field_after = np.linalg.norm(residual[:, :3], axis=1)
    gradient_before = np.abs(source_state[:, 3])
    gradient_after = np.abs(residual[:, 3])
    before_field_rms = float(np.sqrt(np.mean(field_before ** 2)))
    after_field_rms = float(np.sqrt(np.mean(field_after ** 2)))
    before_gradient_rms = float(np.sqrt(np.mean(gradient_before ** 2)))
    after_gradient_rms = float(np.sqrt(np.mean(gradient_after ** 2)))
    result = {
        "field_residual_rms_nt": after_field_rms,
        "field_reduction_percent": 100.0 * (1.0 - after_field_rms / max(before_field_rms, 1.0e-12)),
        "gxx_residual_rms_nt_per_m": after_gradient_rms,
        "gxx_reduction_percent": 100.0 * (
            1.0 - after_gradient_rms / max(before_gradient_rms, 1.0e-12)
        ),
    }
    result.update(
        _gradient_compliance(
            residual,
            config.sample_rate_hz,
            config.gradient_warning_nt_per_m,
            config.gradient_tolerance_nt_per_m,
        )
    )
    result.update(command_metrics(command_a, config.current_limit_a))
    return result


def _target_metrics(
    with_target: np.ndarray,
    counterfactual_no_target: np.ndarray,
    ideal_target: np.ndarray,
    distance_m: np.ndarray,
) -> Dict[str, float]:
    window = distance_m < 12.0
    if window.sum() < 4 or np.ptp(ideal_target[window]) < 1.0e-10:
        return {"transfer_ratio": 0.0, "waveform_correlation": 0.0, "ideal_peak_to_peak_nt": 0.0}
    transferred = with_target[window] - counterfactual_no_target[window]
    reference = ideal_target[window]
    ratio = np.ptp(transferred) / max(np.ptp(reference), 1.0e-12)
    correlation = np.corrcoef(transferred, reference)[0, 1]
    return {
        "transfer_ratio": float(ratio),
        "waveform_correlation": float(correlation),
        "ideal_peak_to_peak_nt": float(np.ptp(reference)),
    }


def _simulate_mode(
    mode: str,
    controller: FeedforwardController,
    predicted_source: np.ndarray,
    true_source: np.ndarray,
    temperature_c: np.ndarray,
    physical_coils: CoilSystem,
    config: SimulationConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    command = controller.commands(predicted_source, temperature_c, mode)
    if config.idealized_hardware:
        actual_current = command.copy()
        # In the present idealized study the plant is exactly the identified
        # response: no current limit, driver lag, coil drift or response error.
        coil_state = controller.coil_model.predict(actual_current, temperature_c)
    else:
        actual_current = simulate_driver(
            command,
            1.0 / config.sample_rate_hz,
            config.coil_driver_tau_s,
            config.current_limit_a,
        )
        coil_state = physical_coils.response_state(actual_current, temperature_c)
    return command, actual_current, true_source + coil_state


def _persist(
    root: Path,
    config: SimulationConfig,
    summary: Dict[str, object],
    time_s: np.ndarray,
    traces: Dict[str, np.ndarray],
) -> Tuple[str, str]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (run_dir / "traces.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        names = list(traces)
        writer.writerow(["time_s"] + names)
        for row in range(len(time_s)):
            writer.writerow([time_s[row]] + [traces[name][row] for name in names])
    return run_id, str(run_dir.resolve())


def run_experiment(config: SimulationConfig, repository_root: Path = None) -> Dict[str, object]:
    root = Path(repository_root or Path(__file__).resolve().parents[1])
    effective_config = config
    if config.idealized_hardware:
        effective_config = replace(
            config,
            bench_sensor_noise_nt=0.0,
            bench_position_error_mm=0.0,
            bench_axis_error_deg=0.0,
            bench_gain_error_fraction=0.0,
            cesium_noise_nt=0.0,
            coil_drift_fraction=0.0,
            source_drift_fraction=0.0,
            post_alignment_source_drift_fraction=0.0,
            post_alignment_coil_drift_fraction=0.0,
        )
    calibration = run_bench_calibration(effective_config)
    profile = generate_navigation_profile(
        config.navigation_duration_s,
        config.sample_rate_hz,
        config.target_enabled,
        config.target_moment_am2,
        config.target_cross_track_m,
        config.target_depth_offset_m,
        config.high_maneuver_enabled,
    )
    dt_s = 1.0 / config.sample_rate_hz
    source_system = MagneticSourceSystem(
        0.0
        if config.idealized_hardware
        else config.source_drift_fraction
        + config.post_alignment_source_drift_fraction,
        source_positions_m=config.source_positions_m(),
        strength_scale=config.source_strength_scale,
    )
    true_source = source_system.field_and_gradient_state(
        profile.source_states, profile.temperature_c, dt_s
    )
    source_contributions = source_system.field_contributions_at_points(
        profile.source_states,
        profile.temperature_c,
        dt_s,
        np.zeros((1, 3), dtype=float),
    )
    navigation_source_features = calibration.source_model.features(
        profile.source_states, profile.temperature_c
    )
    base_predicted_source = calibration.source_model.predict(
        navigation_source_features, config.residual_correction_enabled
    )
    navigation_alignment_features, _ = build_source_alignment_features(
        profile.source_states, profile.temperature_c, config.sample_rate_hz
    )
    predicted_source = calibration.source_alignment.predict(
        navigation_alignment_features, base_predicted_source
    )
    physical_coils = CoilSystem(
        0.0
        if config.idealized_hardware
        else config.coil_drift_fraction + config.post_alignment_coil_drift_fraction,
        uniform_gain_scale=config.uniform_coil_gain_scale,
        gradient_gain_scale=config.gradient_coil_gain_scale,
    )
    controller_current_limit = np.inf if config.idealized_hardware else config.current_limit_a
    controller_slew = np.inf if config.idealized_hardware else config.current_slew_a_per_s
    controller = FeedforwardController(
        calibration.coil_model,
        config.controller_regularization,
        config.gradient_weight,
        controller_current_limit,
        controller_slew,
        dt_s,
        0.0
        if config.idealized_hardware
        else config.coil_driver_tau_s
        * (1.0 + config.driver_tau_model_error_fraction),
        False
        if config.idealized_hardware
        else config.driver_preemphasis_enabled,
    )

    modes = {}
    for mode in ["none", "uniform", "joint_main", "joint"]:
        command, actual_current, residual_state = _simulate_mode(
            mode,
            controller,
            predicted_source,
            true_source,
            profile.temperature_c,
            physical_coils,
            config,
        )
        modes[mode] = {
            "command": command,
            "actual_current": actual_current,
            "residual_state": residual_state,
            "coil_state": residual_state - true_source,
        }

    rng = np.random.RandomState(config.seed + 909)
    white_noise = rng.normal(size=len(profile.time_s))
    random_walk = np.cumsum(rng.normal(size=len(profile.time_s))) * np.sqrt(dt_s)
    if config.idealized_hardware:
        white_noise = np.zeros_like(white_noise)
        random_walk = np.zeros_like(random_walk)
    environment_norm = np.linalg.norm(profile.environment_body_nt, axis=1)
    ideal_target_scalar = (
        np.linalg.norm(profile.environment_body_nt + profile.target_body_nt, axis=1)
        - environment_norm
    )
    ideal_target_sensor = apply_cesium_scalar_sensor(
        environment_norm + ideal_target_scalar,
        dt_s,
        config.cesium_bandwidth_hz,
        0.0,
        np.zeros_like(white_noise),
        np.zeros_like(random_walk),
    ) - environment_norm

    mode_metrics = {}
    cesium_scalar = {}
    counterfactual_scalar = {}
    for mode, values in modes.items():
        local_vector = values["residual_state"][:, :3]
        total_vector = profile.environment_body_nt + profile.target_body_nt + local_vector
        no_target_vector = profile.environment_body_nt + local_vector
        total_magnitude = np.linalg.norm(total_vector, axis=1)
        no_target_magnitude = np.linalg.norm(no_target_vector, axis=1)
        cesium_scalar[mode] = apply_cesium_scalar_sensor(
            total_magnitude,
            dt_s,
            config.cesium_bandwidth_hz,
            effective_config.cesium_noise_nt,
            white_noise,
            random_walk,
        )
        counterfactual_scalar[mode] = apply_cesium_scalar_sensor(
            no_target_magnitude,
            dt_s,
            config.cesium_bandwidth_hz,
            effective_config.cesium_noise_nt,
            white_noise,
            random_walk,
        )
        metric_config = config
        if config.idealized_hardware:
            metric_config = replace(config, current_limit_a=np.inf)
        mode_metrics[mode] = _residual_metrics(
            true_source, values["coil_state"], values["command"], metric_config
        )
        mode_metrics[mode]["cesium_error_rms_nt"] = float(
            np.sqrt(np.mean((cesium_scalar[mode] - (environment_norm + ideal_target_sensor)) ** 2))
        )

    target_metrics = _target_metrics(
        cesium_scalar["joint"],
        counterfactual_scalar["joint"],
        ideal_target_sensor,
        profile.target_distance_m,
    )
    holdout = calibration.metrics["source_model"]["combined_holdout"]
    coil_holdout = calibration.metrics["coil_model"]
    preflight_source = calibration.metrics["preflight_source_alignment"]
    preflight_coil = calibration.metrics["preflight_coil_alignment"]
    source_prediction_error = predicted_source - true_source
    base_source_prediction_error = base_predicted_source - true_source
    equivalent_thruster_current_a = (
        np.abs(profile.source_states[:, 0]) * config.thruster_reference_current_a
    )
    equivalent_thruster_slew_a_per_s = np.abs(
        np.diff(equivalent_thruster_current_a, prepend=equivalent_thruster_current_a[0])
    ) * config.sample_rate_hz
    training_min = np.asarray(calibration.manifest["training_state_min"], dtype=float)
    training_max = np.asarray(calibration.manifest["training_state_max"], dtype=float)
    training_slew = np.asarray(
        calibration.manifest["training_state_slew_max_per_s"], dtype=float
    )
    navigation_min = profile.source_states.min(axis=0)
    navigation_max = profile.source_states.max(axis=0)
    navigation_slew = np.max(
        np.abs(np.diff(profile.source_states, axis=0)) * config.sample_rate_hz,
        axis=0,
    )
    range_covered = (navigation_min >= training_min - 1.0e-9) & (
        navigation_max <= training_max + 1.0e-9
    )
    slew_covered = navigation_slew <= training_slew * 1.001 + 1.0e-9
    excitation_coverage = {
        "range_covered_by_channel": range_covered.tolist(),
        "slew_covered_by_channel": slew_covered.tolist(),
        "all_channels_covered": bool(np.all(range_covered & slew_covered)),
        "navigation_state_min": navigation_min.tolist(),
        "navigation_state_max": navigation_max.tolist(),
        "navigation_state_slew_max_per_s": navigation_slew.tolist(),
        "training_state_min": training_min.tolist(),
        "training_state_max": training_max.tolist(),
        "training_state_slew_max_per_s": training_slew.tolist(),
    }
    calibration.manifest["navigation_excitation_coverage"] = excitation_coverage
    summary = {
        "source_holdout_field_rmse_nt": holdout["field_rmse_nt"],
        "source_holdout_gradient_rmse_nt_per_m": holdout["gradient_rmse_nt_per_m"],
        "coil_holdout_field_rmse_nt": coil_holdout["field_rmse_nt"],
        "coil_holdout_gradient_rmse_nt_per_m": coil_holdout["gradient_rmse_nt_per_m"],
        "preflight_source_holdout": preflight_source,
        "preflight_coil_holdout": preflight_coil,
        "mode_metrics": mode_metrics,
        "target_protection": target_metrics,
        "navigation_prediction": {
            "field_vector_rmse_nt": float(
                np.sqrt(np.mean(np.sum(source_prediction_error[:, :3] ** 2, axis=1)))
            ),
            "gxx_rmse_nt_per_m": float(
                np.sqrt(np.mean(source_prediction_error[:, 3] ** 2))
            ),
            "field_bias_nt": np.mean(source_prediction_error[:, :3], axis=0).tolist(),
            "gxx_bias_nt_per_m": float(np.mean(source_prediction_error[:, 3])),
            "before_alignment_field_vector_rmse_nt": float(
                np.sqrt(np.mean(np.sum(base_source_prediction_error[:, :3] ** 2, axis=1)))
            ),
            "before_alignment_gxx_rmse_nt_per_m": float(
                np.sqrt(np.mean(base_source_prediction_error[:, 3] ** 2))
            ),
        },
        "source_model": calibration.source_model.summary(),
        "preflight_source_alignment": calibration.source_alignment.summary(),
        "field_accuracy_goal": {
            "threshold_rms_nt": 1.0,
            "joint_residual_rms_nt": mode_metrics["joint"]["field_residual_rms_nt"],
            "passed": mode_metrics["joint"]["field_residual_rms_nt"] < 1.0,
        },
        "maneuver": {
            "high_maneuver_enabled": config.high_maneuver_enabled,
            "peak_equivalent_thruster_current_a": float(
                np.max(equivalent_thruster_current_a)
            ),
            "peak_equivalent_thruster_slew_a_per_s": float(
                np.max(equivalent_thruster_slew_a_per_s)
            ),
            "time_at_or_above_reference_current_s": float(
                np.sum(equivalent_thruster_current_a >= config.thruster_reference_current_a)
                / config.sample_rate_hz
            ),
            "excitation_coverage": excitation_coverage,
        },
        "gradient_protection": {
            "warning_nt_per_m": config.gradient_warning_nt_per_m,
            "tolerance_nt_per_m": config.gradient_tolerance_nt_per_m,
            "uncompensated": mode_metrics["none"],
            "joint_compensated": mode_metrics["joint"],
        },
        "assumptions": {
            "scenario_preset": config.scenario_preset,
            "idealized_hardware": config.idealized_hardware,
            "current_limit_applied": not config.idealized_hardware,
            "driver_dynamics_applied": not config.idealized_hardware,
            "coil_response_mismatch_applied": not config.idealized_hardware,
            "preflight_alignment_enabled": config.preflight_alignment_enabled,
            "driver_preemphasis_enabled": bool(
                config.driver_preemphasis_enabled and not config.idealized_hardware
            ),
            "sensor_noise_applied": not config.idealized_hardware,
            "source_drift_fraction": config.source_drift_fraction,
            "coil_drift_fraction": config.coil_drift_fraction,
            "post_alignment_source_drift_fraction": (
                0.0 if config.idealized_hardware else config.post_alignment_source_drift_fraction
            ),
            "post_alignment_coil_drift_fraction": (
                0.0 if config.idealized_hardware else config.post_alignment_coil_drift_fraction
            ),
            "high_maneuver_enabled": config.high_maneuver_enabled,
            "source_strength_scale": config.source_strength_scale,
            "thruster_reference_current_a": config.thruster_reference_current_a,
            "uniform_coil_gain_scale": config.uniform_coil_gain_scale,
            "gradient_coil_gain_scale": config.gradient_coil_gain_scale,
            "bench_sensor_noise_nt": effective_config.bench_sensor_noise_nt,
            "bench_position_error_mm": effective_config.bench_position_error_mm,
            "bench_axis_error_deg": effective_config.bench_axis_error_deg,
            "bench_gain_error_fraction": effective_config.bench_gain_error_fraction,
            "cesium_noise_nt": effective_config.cesium_noise_nt,
            "current_limit_a": None if config.idealized_hardware else config.current_limit_a,
            "driver_tau_s": None if config.idealized_hardware else config.coil_driver_tau_s,
            "driver_tau_model_error_fraction": (
                0.0 if config.idealized_hardware else config.driver_tau_model_error_fraction
            ),
            "current_slew_a_per_s": None
            if config.idealized_hardware
            else config.current_slew_a_per_s,
        },
    }

    bench_count = len(calibration.previews["source_observed"])
    bench_x = np.arange(bench_count, dtype=float) / config.sample_rate_hz
    source_observed = calibration.previews["source_observed"]
    source_predicted_preview = calibration.previews["source_predicted"]
    residual_norms = {
        mode: np.linalg.norm(values["residual_state"][:, :3], axis=1)
        for mode, values in modes.items()
    }
    gxx_residual = {mode: values["residual_state"][:, 3] for mode, values in modes.items()}
    principal_gradient = {
        mode: _principal_gradient_magnitude(values["residual_state"])
        for mode, values in modes.items()
    }
    nav_self_norm = np.linalg.norm(true_source[:, :3], axis=1)
    nav_base_predicted_norm = np.linalg.norm(base_predicted_source[:, :3], axis=1)
    nav_predicted_norm = np.linalg.norm(predicted_source[:, :3], axis=1)
    cesium_anomaly = {mode: values - environment_norm for mode, values in cesium_scalar.items()}
    scene_index = _downsample_indices(len(profile.time_s), config.max_plot_points)
    source_vectors_at_sensor = source_contributions[:, :, 0, :]
    source_positions = {
        source.name: source.position_m.round(7).tolist() for source in source_system.sources
    }
    joint_coil_vector = modes["joint"]["coil_state"][:, :3]
    joint_residual_vector = modes["joint"]["residual_state"][:, :3]
    stabilized_sensor_error = cesium_anomaly["joint"] - ideal_target_sensor

    response = {
        "config": config.to_dict(),
        "summary": summary,
        "manifest": calibration.manifest,
        "labels": {
            "state_vector": STATE_VECTOR_LABELS,
            "source_states": SOURCE_STATE_LABELS,
            "coil_channels": COIL_CHANNEL_LABELS,
        },
        "geometry": {
            "array_positions_m": calibration.array_positions_m.tolist(),
            "source_positions_m": source_positions,
            "cesium_position_m": [0.0, 0.0, 0.0],
            "body_axis_convention": "+X forward, +Y starboard, +Z upward",
            "target_world_m": [0.0, config.target_cross_track_m, -20.0 + config.target_depth_offset_m],
        },
        "spatial_scene": {
            "time_s": profile.time_s[scene_index].round(7).tolist(),
            "source_positions_m": source_positions,
            "source_states": {
                SOURCE_STATE_LABELS[index]: profile.source_states[scene_index, index]
                .round(7)
                .tolist()
                for index in range(len(SOURCE_STATE_LABELS))
            },
            "source_field_vectors_nt": {
                SOURCE_STATE_LABELS[index]: source_vectors_at_sensor[scene_index, index]
                .round(7)
                .tolist()
                for index in range(len(SOURCE_STATE_LABELS))
            },
            "true_total_vector_nt": true_source[scene_index, :3].round(7).tolist(),
            "predicted_total_vector_nt": predicted_source[scene_index, :3].round(7).tolist(),
            "coil_vector_nt": joint_coil_vector[scene_index].round(7).tolist(),
            "residual_vector_nt": joint_residual_vector[scene_index].round(7).tolist(),
            "stabilized_sensor_error_nt": stabilized_sensor_error[scene_index]
            .round(7)
            .tolist(),
            "ideal_target_anomaly_nt": ideal_target_sensor[scene_index].round(7).tolist(),
        },
        "plots": {
            "bench_source_holdout": _line_payload(
                bench_x,
                {
                    "observed_field_norm": np.linalg.norm(source_observed[:, :3], axis=1),
                    "predicted_field_norm": np.linalg.norm(source_predicted_preview[:, :3], axis=1),
                    "observed_gxx": source_observed[:, 3],
                    "predicted_gxx": source_predicted_preview[:, 3],
                },
                config.max_plot_points,
            ),
            "source_states": _line_payload(
                profile.time_s,
                {SOURCE_STATE_LABELS[i]: profile.source_states[:, i] for i in range(4)},
                config.max_plot_points,
            ),
            "maneuver_current": _line_payload(
                profile.time_s,
                {
                    "thruster_equivalent_current_a": equivalent_thruster_current_a,
                    "reference_20a": np.full(
                        len(profile.time_s), config.thruster_reference_current_a
                    ),
                },
                config.max_plot_points,
            ),
            "navigation_model": _line_payload(
                profile.time_s,
                {
                    "true_self_field_norm": nav_self_norm,
                    "base_predicted_self_field_norm": nav_base_predicted_norm,
                    "predicted_self_field_norm": nav_predicted_norm,
                },
                config.max_plot_points,
            ),
            "field_residual": _line_payload(
                profile.time_s, residual_norms, config.max_plot_points
            ),
            "gxx_residual": _line_payload(
                profile.time_s, gxx_residual, config.max_plot_points
            ),
            "gradient_risk": _line_payload(
                profile.time_s,
                {
                    "uncompensated": principal_gradient["none"],
                    "main_gxx_only": principal_gradient["joint_main"],
                    "joint_compensated": principal_gradient["joint"],
                    "warning_656": np.full(
                        len(profile.time_s), config.gradient_warning_nt_per_m
                    ),
                    "tolerance_10000": np.full(
                        len(profile.time_s), config.gradient_tolerance_nt_per_m
                    ),
                },
                config.max_plot_points,
            ),
            "coil_currents": _line_payload(
                profile.time_s,
                {
                    COIL_CHANNEL_LABELS[i]: modes["joint"]["actual_current"][:, i]
                    for i in range(len(COIL_CHANNEL_LABELS))
                },
                config.max_plot_points,
            ),
            "cesium_anomaly": _line_payload(
                profile.time_s,
                dict(cesium_anomaly, ideal_target=ideal_target_sensor),
                config.max_plot_points,
            ),
            "trajectory": _line_payload(
                profile.position_world_m[:, 0],
                {
                    "auv_y": profile.position_world_m[:, 1],
                    "target_y": np.full(len(profile.time_s), config.target_cross_track_m),
                    "target_distance": profile.target_distance_m,
                },
                config.max_plot_points,
            ),
        },
        "run_id": None,
        "run_directory": None,
    }

    if config.persist_run:
        trace_export = {
            "self_field_true_norm_nt": nav_self_norm,
            "self_field_predicted_norm_nt": nav_predicted_norm,
            "residual_none_norm_nt": residual_norms["none"],
            "residual_uniform_norm_nt": residual_norms["uniform"],
            "residual_joint_norm_nt": residual_norms["joint"],
            "gxx_none_nt_per_m": gxx_residual["none"],
            "gxx_joint_nt_per_m": gxx_residual["joint"],
            "principal_gradient_none_nt_per_m": principal_gradient["none"],
            "principal_gradient_joint_nt_per_m": principal_gradient["joint"],
            "thruster_equivalent_current_a": equivalent_thruster_current_a,
            "cesium_joint_nt": cesium_scalar["joint"],
            "ideal_target_anomaly_nt": ideal_target_sensor,
        }
        run_id, run_dir = _persist(root, config, summary, profile.time_s, trace_export)
        response["run_id"] = run_id
        response["run_directory"] = run_dir
    return response
