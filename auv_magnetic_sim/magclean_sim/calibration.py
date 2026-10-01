from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .coils import CoilSystem, generate_coil_calibration_currents
from .config import SimulationConfig
from .features import build_source_alignment_features
from .models import (
    CoilResponseModel,
    PreflightSourceAlignment,
    SourceFieldModel,
    component_metrics,
)
from .sensors import BenchVectorArray
from .sources import MagneticSourceSystem, generate_scenario_states


BASE_SCENARIOS = [
    "thruster_sweep",
    "dc_dc_steps",
    "actuator_cycle",
    "electronics_cycle",
    "thruster_dc_dc",
    "thruster_actuator",
    "dc_electronics_train",
    "actuator_electronics_train",
    "random_multisine_train",
    "thermal_mixed_train",
    "load_transients_train",
    "high_maneuver_train",
    "rapid_reversal_train",
]
CORRECTION_SCENARIOS = ["correction_mixed", "correction_high_dynamic"]
VALIDATION_SCENARIOS = [
    "holdout_converter_actuator",
    "holdout_all_sources",
    "holdout_high_maneuver",
]
PREFLIGHT_SOURCE_TRAIN_SCENARIOS = [
    "preflight_anchor_train_a",
    "preflight_anchor_train_b",
]
PREFLIGHT_SOURCE_VALIDATION_SCENARIOS = ["preflight_anchor_holdout"]


@dataclass
class CalibrationBundle:
    source_model: SourceFieldModel
    source_alignment: PreflightSourceAlignment
    coil_model: CoilResponseModel
    metrics: Dict[str, object]
    previews: Dict[str, object]
    manifest: Dict[str, object]
    array_positions_m: np.ndarray


def _source_block(
    scenario: str,
    config: SimulationConfig,
    forward: MagneticSourceSystem,
    sensor: BenchVectorArray,
    source_model: SourceFieldModel,
    rng: np.random.RandomState,
    sample_count: int = None,
) -> Dict[str, np.ndarray]:
    states, temperature = generate_scenario_states(
        scenario,
        sample_count or config.bench_samples_per_scenario,
        config.sample_rate_hz,
        rng,
    )
    fields = forward.field_at_points(
        states, temperature, 1.0 / config.sample_rate_hz, sensor.true_positions_m
    )
    _, reconstructed = sensor.observe_fields(fields, rng)
    features = source_model.features(states, temperature)
    return {
        "states": states,
        "temperature": temperature,
        "observed": reconstructed,
        "features": features,
    }


def run_bench_calibration(config: SimulationConfig) -> CalibrationBundle:
    rng = np.random.RandomState(config.seed + 101)
    sensor = BenchVectorArray(
        noise_nt=config.bench_sensor_noise_nt,
        rng=rng,
        position_error_mm=config.bench_position_error_mm,
        axis_error_deg=config.bench_axis_error_deg,
        gain_error_fraction=config.bench_gain_error_fraction,
    )
    source_forward = MagneticSourceSystem(
        drift_fraction=0.0,
        source_positions_m=config.source_positions_m(),
        strength_scale=config.source_strength_scale,
    )
    source_model = SourceFieldModel(
        ridge=config.source_model_ridge,
        residual_limit=config.residual_correction_limit,
        sample_rate_hz=config.sample_rate_hz,
        variant=config.source_model_variant,
    )

    source_blocks = {}
    for scenario in BASE_SCENARIOS + CORRECTION_SCENARIOS + VALIDATION_SCENARIOS:
        source_blocks[scenario] = _source_block(
            scenario, config, source_forward, sensor, source_model, rng
        )
    base_x = np.vstack([source_blocks[name]["features"] for name in BASE_SCENARIOS])
    base_y = np.vstack([source_blocks[name]["observed"] for name in BASE_SCENARIOS])
    source_model.fit_base(base_x, base_y)
    correction_x = np.vstack(
        [source_blocks[name]["features"] for name in CORRECTION_SCENARIOS]
    )
    correction_y = np.vstack(
        [source_blocks[name]["observed"] for name in CORRECTION_SCENARIOS]
    )
    if config.residual_correction_enabled:
        source_model.fit_correction(correction_x, correction_y)

    source_metrics = {}
    preview_observed = []
    preview_predicted = []
    preview_scenario = []
    for scenario in VALIDATION_SCENARIOS:
        block = source_blocks[scenario]
        predicted = source_model.predict(block["features"])
        source_metrics[scenario] = component_metrics(block["observed"], predicted)
        preview_observed.append(block["observed"])
        preview_predicted.append(predicted)
        preview_scenario.extend([scenario] * len(predicted))
    all_source_observed = np.vstack(preview_observed)
    all_source_predicted = np.vstack(preview_predicted)
    source_metrics["combined_holdout"] = component_metrics(
        all_source_observed, all_source_predicted
    )

    # Independent commissioning transfer stage.  These excitation families
    # are disjoint from base training, residual correction, formal holdout and
    # navigation.  They see the present (drifted) platform only while the
    # removable vector array is still installed.
    source_alignment = PreflightSourceAlignment(config.preflight_source_ridge)
    preflight_source_metrics = {"enabled": config.preflight_alignment_enabled}
    preflight_source_observed = np.empty((0, 8), dtype=float)
    preflight_source_predicted = np.empty((0, 8), dtype=float)
    if config.preflight_alignment_enabled:
        preflight_rng = np.random.RandomState(config.seed + 303)
        transfer_forward = MagneticSourceSystem(
            drift_fraction=config.source_drift_fraction,
            source_positions_m=config.source_positions_m(),
            strength_scale=config.source_strength_scale,
        )
        transfer_blocks = {}
        for scenario in (
            PREFLIGHT_SOURCE_TRAIN_SCENARIOS
            + PREFLIGHT_SOURCE_VALIDATION_SCENARIOS
        ):
            transfer_blocks[scenario] = _source_block(
                scenario,
                config,
                transfer_forward,
                sensor,
                source_model,
                preflight_rng,
                config.preflight_samples_per_scenario,
            )
            block = transfer_blocks[scenario]
            nominal_fields = source_forward.field_at_points(
                block["states"],
                block["temperature"],
                1.0 / config.sample_rate_hz,
                sensor.true_positions_m,
            )
            _, block["nominal_observed"] = sensor.observe_fields(
                nominal_fields, preflight_rng
            )
            block["alignment_features"], _ = build_source_alignment_features(
                block["states"], block["temperature"], config.sample_rate_hz
            )
        transfer_train_features = np.vstack(
            [
                transfer_blocks[name]["alignment_features"]
                for name in PREFLIGHT_SOURCE_TRAIN_SCENARIOS
            ]
        )
        transfer_train_delta = np.vstack(
            [
                transfer_blocks[name]["observed"]
                - transfer_blocks[name]["nominal_observed"]
                for name in PREFLIGHT_SOURCE_TRAIN_SCENARIOS
            ]
        )
        transfer_train_base = np.vstack(
            [
                source_model.predict(transfer_blocks[name]["features"])
                for name in PREFLIGHT_SOURCE_TRAIN_SCENARIOS
            ]
        )
        source_alignment.fit(
            transfer_train_features,
            transfer_train_base,
            transfer_train_base + transfer_train_delta,
        )
        transfer_holdout = transfer_blocks[PREFLIGHT_SOURCE_VALIDATION_SCENARIOS[0]]
        transfer_holdout_base = source_model.predict(transfer_holdout["features"])
        preflight_source_predicted = source_alignment.predict(
            transfer_holdout["alignment_features"], transfer_holdout_base
        )
        preflight_source_observed = transfer_holdout["observed"]
        transfer_holdout_delta = (
            transfer_holdout["observed"] - transfer_holdout["nominal_observed"]
        )
        predicted_holdout_delta = preflight_source_predicted - transfer_holdout_base
        preflight_source_metrics.update(
            {
                "before_alignment": component_metrics(
                    preflight_source_observed, transfer_holdout_base
                ),
                "after_alignment": component_metrics(
                    preflight_source_observed, preflight_source_predicted
                ),
                "drift_delta_before_alignment": component_metrics(
                    transfer_holdout_delta, np.zeros_like(transfer_holdout_delta)
                ),
                "drift_delta_after_alignment": component_metrics(
                    transfer_holdout_delta, predicted_holdout_delta
                ),
            }
        )

    coil_forward = CoilSystem(
        drift_fraction=0.0,
        uniform_gain_scale=config.uniform_coil_gain_scale,
        gradient_gain_scale=config.gradient_coil_gain_scale,
    )
    currents, coil_temperature, coil_labels = generate_coil_calibration_currents(
        config.bench_samples_per_scenario,
        config.sample_rate_hz,
        rng,
        config.current_limit_a,
    )
    coil_fields = coil_forward.array_field(
        currents, coil_temperature, sensor.true_positions_m
    )
    _, coil_observed = sensor.observe_fields(coil_fields, rng)
    coil_train = coil_labels != "holdout_multichannel"
    coil_holdout = ~coil_train
    nominal_coil_model = CoilResponseModel(config.coil_model_ridge)
    nominal_coil_model.fit(
        currents[coil_train], coil_temperature[coil_train], coil_observed[coil_train]
    )
    coil_predicted = nominal_coil_model.predict(
        currents[coil_holdout], coil_temperature[coil_holdout]
    )
    coil_metrics = component_metrics(coil_observed[coil_holdout], coil_predicted)

    coil_model = nominal_coil_model
    preflight_coil_metrics = {"enabled": config.preflight_alignment_enabled}
    if config.preflight_alignment_enabled:
        preflight_coil_rng = np.random.RandomState(config.seed + 404)
        transfer_coil_forward = CoilSystem(
            drift_fraction=config.coil_drift_fraction,
            uniform_gain_scale=config.uniform_coil_gain_scale,
            gradient_gain_scale=config.gradient_coil_gain_scale,
        )
        transfer_currents, transfer_temperature, transfer_labels = (
            generate_coil_calibration_currents(
                config.preflight_samples_per_scenario,
                config.sample_rate_hz,
                preflight_coil_rng,
                config.current_limit_a,
            )
        )
        transfer_fields = transfer_coil_forward.array_field(
            transfer_currents, transfer_temperature, sensor.true_positions_m
        )
        _, transfer_coil_observed = sensor.observe_fields(
            transfer_fields, preflight_coil_rng
        )
        transfer_coil_train = transfer_labels != "holdout_multichannel"
        transfer_coil_holdout = ~transfer_coil_train
        coil_model = CoilResponseModel(config.coil_model_ridge)
        coil_model.fit(
            transfer_currents[transfer_coil_train],
            transfer_temperature[transfer_coil_train],
            transfer_coil_observed[transfer_coil_train],
        )
        transfer_coil_predicted = coil_model.predict(
            transfer_currents[transfer_coil_holdout],
            transfer_temperature[transfer_coil_holdout],
        )
        preflight_coil_metrics["holdout"] = component_metrics(
            transfer_coil_observed[transfer_coil_holdout], transfer_coil_predicted
        )

    base_states = np.vstack([source_blocks[name]["states"] for name in BASE_SCENARIOS])
    base_slew = np.vstack(
        [
            np.abs(np.diff(source_blocks[name]["states"], axis=0))
            * config.sample_rate_hz
            for name in BASE_SCENARIOS
        ]
    )
    return CalibrationBundle(
        source_model=source_model,
        source_alignment=source_alignment,
        coil_model=coil_model,
        metrics={
            "source_model": source_metrics,
            "coil_model": coil_metrics,
            "preflight_source_alignment": preflight_source_metrics,
            "preflight_coil_alignment": preflight_coil_metrics,
        },
        previews={
            "source_observed": all_source_observed,
            "source_predicted": all_source_predicted,
            "source_scenarios": preview_scenario,
            "coil_observed": coil_observed[coil_holdout],
            "coil_predicted": coil_predicted,
            "coil_currents": currents[coil_holdout],
            "preflight_source_observed": preflight_source_observed,
            "preflight_source_predicted": preflight_source_predicted,
        },
        manifest={
            "base_training_scenarios": list(BASE_SCENARIOS),
            "residual_correction_scenarios": list(CORRECTION_SCENARIOS),
            "holdout_validation_scenarios": list(VALIDATION_SCENARIOS),
            "preflight_source_training_scenarios": list(
                PREFLIGHT_SOURCE_TRAIN_SCENARIOS
            ),
            "preflight_source_validation_scenarios": list(
                PREFLIGHT_SOURCE_VALIDATION_SCENARIOS
            ),
            "preflight_alignment_enabled": config.preflight_alignment_enabled,
            "preflight_samples_per_scenario": config.preflight_samples_per_scenario,
            "preflight_source_alignment": source_alignment.summary(),
            "coil_training": "eight disjoint single-channel sweeps",
            "coil_holdout": "unseen simultaneous eight-channel excitation",
            "preflight_coil_training": "new eight-channel single-axis commissioning sweeps",
            "preflight_coil_holdout": "new unseen simultaneous commissioning excitation",
            "navigation_array_installed": False,
            "model_update_during_navigation": False,
            "cesium_used_for_feedback": False,
            "source_model_variant": config.source_model_variant,
            "source_model_feature_count": len(source_model.feature_names),
            "source_strength_scale": config.source_strength_scale,
            "training_state_min": base_states.min(axis=0).tolist(),
            "training_state_max": base_states.max(axis=0).tolist(),
            "training_state_slew_max_per_s": base_slew.max(axis=0).tolist(),
            "array_geometry": "7-point symmetric central-difference fixture, 10 mm half-baseline",
            "coil_authority_scale": {
                "uniform": config.uniform_coil_gain_scale,
                "gradient": config.gradient_coil_gain_scale,
            },
            "idealized_hardware": config.idealized_hardware,
            "bench_error_model": {
                "white_noise_nt": config.bench_sensor_noise_nt,
                "position_sigma_mm": config.bench_position_error_mm,
                "axis_sigma_deg": config.bench_axis_error_deg,
                "gain_sigma_fraction": config.bench_gain_error_fraction,
            },
        },
        array_positions_m=sensor.positions_m,
    )
