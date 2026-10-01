from pathlib import Path

import numpy as np

from magclean_sim.calibration import (
    BASE_SCENARIOS,
    CORRECTION_SCENARIOS,
    PREFLIGHT_SOURCE_TRAIN_SCENARIOS,
    PREFLIGHT_SOURCE_VALIDATION_SCENARIOS,
    VALIDATION_SCENARIOS,
)
from magclean_sim.config import SimulationConfig
from magclean_sim.experiment import run_experiment
from magclean_sim.math3d import reconstruct_state8, state8_to_array_field
from magclean_sim.sensors import DEFAULT_ARRAY_POSITIONS_M


def test_array_reconstruction_is_exact_without_noise():
    state = np.array([12.0, -8.0, 33.0, 40.0, -17.0, 4.0, -6.0, 9.0])
    fields = state8_to_array_field(state, DEFAULT_ARRAY_POSITIONS_M)
    reconstructed = reconstruct_state8(fields, DEFAULT_ARRAY_POSITIONS_M)
    assert np.allclose(reconstructed, state, atol=1.0e-7)


def test_calibration_scenario_roles_are_disjoint():
    role_sets = [
        set(BASE_SCENARIOS),
        set(CORRECTION_SCENARIOS),
        set(VALIDATION_SCENARIOS),
        set(PREFLIGHT_SOURCE_TRAIN_SCENARIOS),
        set(PREFLIGHT_SOURCE_VALIDATION_SCENARIOS),
    ]
    for left_index, left in enumerate(role_sets):
        for right in role_sets[left_index + 1 :]:
            assert not (left & right)


def test_joint_compensation_reduces_field_and_main_gradient():
    config = SimulationConfig(navigation_duration_s=40.0, bench_samples_per_scenario=100)
    summary = run_experiment(config)["summary"]
    none = summary["mode_metrics"]["none"]
    joint = summary["mode_metrics"]["joint"]
    assert joint["field_residual_rms_nt"] < 0.5 * none["field_residual_rms_nt"]
    assert joint["gxx_residual_rms_nt_per_m"] < 0.8 * none["gxx_residual_rms_nt_per_m"]


def test_enhanced_estimator_outperforms_legacy_on_navigation():
    common = dict(
        navigation_duration_s=30.0,
        bench_samples_per_scenario=260,
        seed=831,
        idealized_hardware=True,
        source_drift_fraction=0.0,
        coil_drift_fraction=0.0,
    )
    legacy = run_experiment(
        SimulationConfig(source_model_variant="legacy", source_model_ridge=0.02, **common)
    )["summary"]["navigation_prediction"]
    enhanced = run_experiment(
        SimulationConfig(
            source_model_variant="enhanced_dynamic", source_model_ridge=5.0e-7, **common
        )
    )["summary"]["navigation_prediction"]
    assert enhanced["field_vector_rmse_nt"] < 0.05 * legacy["field_vector_rmse_nt"]
    assert enhanced["gxx_rmse_nt_per_m"] < 0.05 * legacy["gxx_rmse_nt_per_m"]


def test_idealized_mode_removes_hardware_limits_and_nearly_nulls_residual():
    result = run_experiment(
        SimulationConfig(
            scenario_preset="ideal",
            navigation_duration_s=30.0,
            bench_samples_per_scenario=260,
            idealized_hardware=True,
            source_model_ridge=5.0e-7,
            source_drift_fraction=0.0,
            coil_drift_fraction=0.0,
        )
    )["summary"]
    joint = result["mode_metrics"]["joint"]
    assert result["assumptions"]["current_limit_applied"] is False
    assert joint["peak_current_a"] > 1.5
    assert joint["saturation_fraction"] == 0.0
    assert joint["field_residual_rms_nt"] < 0.1
    assert joint["gxx_residual_rms_nt_per_m"] < 0.25


def test_target_does_not_change_feedforward_performance_metrics():
    base = dict(navigation_duration_s=30.0, bench_samples_per_scenario=80, seed=2201)
    with_target = run_experiment(SimulationConfig(target_enabled=True, **base))
    without_target = run_experiment(SimulationConfig(target_enabled=False, **base))
    for mode in ["none", "uniform", "joint_main", "joint"]:
        left = with_target["summary"]["mode_metrics"][mode]
        right = without_target["summary"]["mode_metrics"][mode]
        # Cesium scalar error legitimately changes because the target is part
        # of the measured physical field. Commands and compensation metrics must not.
        for metric in [
            "field_residual_rms_nt",
            "field_reduction_percent",
            "gxx_residual_rms_nt_per_m",
            "gxx_reduction_percent",
            "peak_current_a",
            "rms_current_a",
            "saturation_fraction",
        ]:
            assert left[metric] == right[metric]


def test_flask_health_and_short_simulation():
    from magclean_sim.webapp import create_app

    app = create_app(Path(__file__).resolve().parents[1])
    client = app.test_client()
    assert client.get("/api/health").status_code == 200
    presets = client.get("/api/presets")
    assert presets.status_code == 200
    assert set(presets.get_json()) == {"ideal", "moderate", "robust", "high_maneuver"}
    response = client.post(
        "/api/simulate",
        json={"navigation_duration_s": 20.0, "bench_samples_per_scenario": 60},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert "summary" in payload
    assert "spatial_scene" in payload
    assert len(payload["spatial_scene"]["time_s"]) > 1
    assert set(payload["spatial_scene"]["source_field_vectors_nt"]) == {
        "thruster",
        "dc_dc",
        "actuator",
        "electronics",
    }


def test_web_source_coordinates_are_applied_to_forward_model_and_scene():
    from magclean_sim.webapp import create_app

    app = create_app(Path(__file__).resolve().parents[1])
    client = app.test_client()
    response = client.post(
        "/api/simulate",
        json={
            "navigation_duration_s": 20.0,
            "bench_samples_per_scenario": 60,
            "thruster_x_m": -1.15,
            "thruster_y_m": 0.18,
            "thruster_z_m": -0.09,
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    expected = [-1.15, 0.18, -0.09]
    assert payload["geometry"]["source_positions_m"]["thruster"] == expected
    assert payload["spatial_scene"]["source_positions_m"]["thruster"] == expected


def test_source_cannot_overlap_cesium_sensitive_volume():
    config = SimulationConfig(thruster_x_m=0.0, thruster_y_m=0.0, thruster_z_m=0.0)
    try:
        config.source_positions_m()
    except ValueError as exc:
        assert "at least 0.05 m" in str(exc)
    else:
        raise AssertionError("Expected source position validation to reject an overlap")


def test_high_maneuver_exceeds_g882_gradient_limit_then_recovers():
    result = run_experiment(
        SimulationConfig(
            scenario_preset="high_maneuver",
            high_maneuver_enabled=True,
            source_strength_scale=15.0,
            uniform_coil_gain_scale=5.0,
            gradient_coil_gain_scale=50.0,
            navigation_duration_s=40.0,
            bench_samples_per_scenario=160,
            max_plot_points=2000,
        )
    )
    summary = result["summary"]
    none = summary["mode_metrics"]["none"]
    main_only = summary["mode_metrics"]["joint_main"]
    joint = summary["mode_metrics"]["joint"]
    assert summary["maneuver"]["peak_equivalent_thruster_current_a"] >= 30.0
    assert summary["maneuver"]["excitation_coverage"]["all_channels_covered"] is True
    assert none["principal_peak_nt_per_m"] > 10000.0
    assert none["tolerance_exceed_duration_s"] > 1.0
    assert main_only["tolerance_exceed_duration_s"] > 1.0
    assert joint["principal_peak_nt_per_m"] < 656.0
    assert joint["tolerance_exceed_duration_s"] == 0.0
    assert joint["field_residual_rms_nt"] < 1.0
    assert summary["field_accuracy_goal"]["passed"] is True
    assert result["manifest"]["navigation_array_installed"] is False
    assert result["manifest"]["model_update_during_navigation"] is False
    assert result["manifest"]["cesium_used_for_feedback"] is False
    assert len(result["labels"]["coil_channels"]) == 8
    assert set(result["plots"]["gradient_risk"]["series"]) == {
        "uncompensated",
        "main_gxx_only",
        "joint_compensated",
        "warning_656",
        "tolerance_10000",
    }
