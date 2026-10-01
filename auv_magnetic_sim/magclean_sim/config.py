from dataclasses import asdict, dataclass, fields
from math import isfinite, sqrt
from typing import Any, Dict


@dataclass
class SimulationConfig:
    """All user-adjustable simulation parameters.

    Values are deliberately kept in one dataclass so experiment manifests can
    be serialized without relying on hidden module globals.
    """

    seed: int = 20260923
    sample_rate_hz: float = 20.0
    navigation_duration_s: float = 120.0
    scenario_preset: str = "high_maneuver"
    bench_samples_per_scenario: int = 600

    bench_sensor_noise_nt: float = 0.08
    bench_position_error_mm: float = 0.18
    bench_axis_error_deg: float = 0.08
    bench_gain_error_fraction: float = 0.0008
    cesium_noise_nt: float = 0.04
    cesium_bandwidth_hz: float = 8.0

    source_model_variant: str = "enhanced_dynamic"
    source_model_ridge: float = 0.003
    coil_model_ridge: float = 0.002
    residual_correction_enabled: bool = True
    residual_correction_limit: float = 0.15

    # A short, scenario-disjoint commissioning calibration is performed with
    # the removable vector array immediately before it is removed for sea
    # operation.  It adapts the frozen bench models to the present source and
    # coil transfer gains; navigation and cesium data are never used.
    preflight_alignment_enabled: bool = True
    preflight_samples_per_scenario: int = 360
    preflight_source_ridge: float = 0.01

    source_drift_fraction: float = 0.005
    coil_drift_fraction: float = 0.005
    # Additional short-term change after the preflight array has been removed.
    # This prevents the commissioning alignment from assuming a perfectly
    # stationary plant throughout the subsequent navigation run.
    post_alignment_source_drift_fraction: float = 0.0003
    post_alignment_coil_drift_fraction: float = 0.0003

    # Stress-test controls. The source scale multiplies the preliminary
    # equivalent magnetic moments; it is deliberately exposed because it is a
    # bench-calibration replacement parameter, not a measured hardware fact.
    high_maneuver_enabled: bool = True
    source_strength_scale: float = 15.0
    thruster_reference_current_a: float = 20.0

    # Candidate coil authority scales relative to the preliminary response
    # matrix. These are design targets until replaced by measured coil data.
    uniform_coil_gain_scale: float = 5.0
    gradient_coil_gain_scale: float = 50.0

    # G-882 engineering thresholds used only for compliance/risk reporting.
    gradient_warning_nt_per_m: float = 656.0
    gradient_tolerance_nt_per_m: float = 10000.0

    # Body-frame coordinates relative to the cesium sensitive volume.
    # Convention: +X forward, +Y starboard, +Z upward; unit: metre.
    thruster_x_m: float = -0.82
    thruster_y_m: float = 0.10
    thruster_z_m: float = -0.12
    dc_dc_x_m: float = -0.38
    dc_dc_y_m: float = -0.24
    dc_dc_z_m: float = 0.13
    actuator_x_m: float = 0.27
    actuator_y_m: float = 0.22
    actuator_z_m: float = -0.17
    electronics_x_m: float = 0.05
    electronics_y_m: float = -0.31
    electronics_z_m: float = 0.08

    idealized_hardware: bool = False
    coil_driver_tau_s: float = 0.015
    driver_tau_model_error_fraction: float = 0.05
    driver_preemphasis_enabled: bool = True
    current_limit_a: float = 8.0
    current_slew_a_per_s: float = 250.0
    controller_regularization: float = 0.02
    gradient_weight: float = 1.0

    target_enabled: bool = True
    target_moment_am2: float = 45.0
    target_cross_track_m: float = 4.0
    target_depth_offset_m: float = 0.5

    persist_run: bool = False
    max_plot_points: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def source_positions_m(self) -> Dict[str, list]:
        """Return validated body-frame source coordinates keyed by source name."""
        positions = {
            "thruster": [self.thruster_x_m, self.thruster_y_m, self.thruster_z_m],
            "dc_dc": [self.dc_dc_x_m, self.dc_dc_y_m, self.dc_dc_z_m],
            "actuator": [self.actuator_x_m, self.actuator_y_m, self.actuator_z_m],
            "electronics": [self.electronics_x_m, self.electronics_y_m, self.electronics_z_m],
        }
        for name, position in positions.items():
            if not all(isfinite(float(value)) for value in position):
                raise ValueError("Source %s position contains a non-finite value" % name)
            distance = sqrt(sum(float(value) ** 2 for value in position))
            if distance < 0.05:
                raise ValueError(
                    "Source %s must be at least 0.05 m from the cesium sensitive volume" % name
                )
        return positions

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "SimulationConfig":
        allowed = {f.name for f in fields(cls)}
        clean = {key: value for key, value in values.items() if key in allowed}
        return cls(**clean)


STATE_VECTOR_LABELS = [
    "Bx_nT",
    "By_nT",
    "Bz_nT",
    "Gxx_nT_per_m",
    "Gyy_nT_per_m",
    "Gxy_nT_per_m",
    "Gxz_nT_per_m",
    "Gyz_nT_per_m",
]

SOURCE_STATE_LABELS = [
    "thruster",
    "dc_dc",
    "actuator",
    "electronics",
]

COIL_CHANNEL_LABELS = [
    "uniform_x",
    "uniform_y",
    "uniform_z",
    "gradient_gxx",
    "gradient_gyy",
    "gradient_gxy",
    "gradient_gxz",
    "gradient_gyz",
]
