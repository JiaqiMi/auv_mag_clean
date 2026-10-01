import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .config import SimulationConfig
from .experiment import run_experiment
from .presets import SIMULATION_PRESETS


METRICS = [
    "prediction_field_rmse_nt",
    "prediction_gxx_rmse_nt_per_m",
    "residual_field_rms_nt",
    "residual_gxx_rms_nt_per_m",
    "field_reduction_percent",
    "gxx_reduction_percent",
    "cesium_error_rms_nt",
    "peak_current_a",
    "saturation_fraction",
    "target_correlation",
    "uncompensated_principal_peak_nt_per_m",
    "uncompensated_tolerance_exceed_duration_s",
    "joint_principal_peak_nt_per_m",
    "joint_tolerance_exceed_duration_s",
    "gradient_valid_fraction",
    "excitation_coverage_pass",
]


def _case(preset_name: str, seed: int) -> Dict[str, object]:
    values = dict(SIMULATION_PRESETS[preset_name]["values"])
    values["seed"] = int(seed)
    summary = run_experiment(SimulationConfig.from_dict(values))["summary"]
    prediction = summary["navigation_prediction"]
    joint = summary["mode_metrics"]["joint"]
    none = summary["mode_metrics"]["none"]
    return {
        "preset": preset_name,
        "seed": int(seed),
        "prediction_field_rmse_nt": prediction["field_vector_rmse_nt"],
        "prediction_gxx_rmse_nt_per_m": prediction["gxx_rmse_nt_per_m"],
        "residual_field_rms_nt": joint["field_residual_rms_nt"],
        "residual_gxx_rms_nt_per_m": joint["gxx_residual_rms_nt_per_m"],
        "field_reduction_percent": joint["field_reduction_percent"],
        "gxx_reduction_percent": joint["gxx_reduction_percent"],
        "cesium_error_rms_nt": joint["cesium_error_rms_nt"],
        "peak_current_a": joint["peak_current_a"],
        "saturation_fraction": joint["saturation_fraction"],
        "target_correlation": summary["target_protection"]["waveform_correlation"],
        "uncompensated_principal_peak_nt_per_m": none["principal_peak_nt_per_m"],
        "uncompensated_tolerance_exceed_duration_s": none[
            "tolerance_exceed_duration_s"
        ],
        "joint_principal_peak_nt_per_m": joint["principal_peak_nt_per_m"],
        "joint_tolerance_exceed_duration_s": joint["tolerance_exceed_duration_s"],
        "gradient_valid_fraction": joint["valid_fraction_by_gradient"],
        "excitation_coverage_pass": float(
            summary["maneuver"]["excitation_coverage"]["all_channels_covered"]
        ),
    }


def _aggregate(rows: List[Dict[str, object]]) -> Dict[str, object]:
    result = {"case_count": len(rows)}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        result[metric] = {
            "mean": float(values.mean()),
            "p95": float(np.percentile(values, 95)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def validate_presets(seeds: Iterable[int]) -> Dict[str, object]:
    seeds = list(seeds)
    cases = [
        _case(preset_name, seed)
        for preset_name in ["ideal", "moderate", "robust", "high_maneuver"]
        for seed in seeds
    ]
    aggregates = {
        name: _aggregate([row for row in cases if row["preset"] == name])
        for name in ["ideal", "moderate", "robust", "high_maneuver"]
    }
    criteria = {
        "ideal": {
            "residual_field_max_le_0_1_nt": aggregates["ideal"]["residual_field_rms_nt"]["max"] <= 0.1,
            "residual_gxx_max_le_0_25_nt_per_m": aggregates["ideal"]["residual_gxx_rms_nt_per_m"]["max"] <= 0.25,
        },
        "moderate": {
            "residual_field_max_le_2_2_nt": aggregates["moderate"]["residual_field_rms_nt"]["max"] <= 2.2,
            "residual_gxx_max_le_12_5_nt_per_m": aggregates["moderate"]["residual_gxx_rms_nt_per_m"]["max"] <= 12.5,
            "field_reduction_min_ge_97_percent": aggregates["moderate"]["field_reduction_percent"]["min"] >= 97.0,
            "gxx_reduction_min_ge_96_percent": aggregates["moderate"]["gxx_reduction_percent"]["min"] >= 96.0,
        },
        "robust": {
            "residual_field_max_le_3_2_nt": aggregates["robust"]["residual_field_rms_nt"]["max"] <= 3.2,
            "residual_gxx_max_le_25_nt_per_m": aggregates["robust"]["residual_gxx_rms_nt_per_m"]["max"] <= 25.0,
            "field_reduction_min_ge_96_percent": aggregates["robust"]["field_reduction_percent"]["min"] >= 96.0,
            "gxx_reduction_min_ge_92_percent": aggregates["robust"]["gxx_reduction_percent"]["min"] >= 92.0,
        },
        "high_maneuver": {
            "residual_field_max_lt_1_nt": aggregates["high_maneuver"]["residual_field_rms_nt"]["max"] < 1.0,
            "cesium_error_max_lt_1_nt": aggregates["high_maneuver"]["cesium_error_rms_nt"]["max"] < 1.0,
            "uncompensated_peak_min_gt_10000_nt_per_m": aggregates["high_maneuver"]["uncompensated_principal_peak_nt_per_m"]["min"] > 10000.0,
            "uncompensated_exceed_duration_min_gt_1_s": aggregates["high_maneuver"]["uncompensated_tolerance_exceed_duration_s"]["min"] > 1.0,
            "joint_peak_max_lt_656_nt_per_m": aggregates["high_maneuver"]["joint_principal_peak_nt_per_m"]["max"] < 656.0,
            "joint_exceed_duration_max_eq_0_s": aggregates["high_maneuver"]["joint_tolerance_exceed_duration_s"]["max"] == 0.0,
            "gradient_valid_fraction_min_eq_1": aggregates["high_maneuver"]["gradient_valid_fraction"]["min"] == 1.0,
            "excitation_coverage_all_cases": aggregates["high_maneuver"]["excitation_coverage_pass"]["min"] == 1.0,
        },
    }
    all_passed = all(value for group in criteria.values() for value in group.values())
    return {
        "seeds": seeds,
        "presets": SIMULATION_PRESETS,
        "aggregates": aggregates,
        "criteria": criteria,
        "all_criteria_passed": all_passed,
        "cases": cases,
    }


def save_preset_validation(report: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preset_validation_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = report["cases"]
    with (output_dir / "preset_validation_cases.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
