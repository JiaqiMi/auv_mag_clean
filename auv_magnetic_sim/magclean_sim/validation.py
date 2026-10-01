import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .config import SimulationConfig
from .experiment import run_experiment


METRICS = [
    "holdout_field_rmse_nt",
    "holdout_gradient_rmse_nt_per_m",
    "navigation_field_rmse_nt",
    "navigation_gxx_rmse_nt_per_m",
    "residual_field_rms_nt",
    "residual_gxx_rms_nt_per_m",
    "peak_required_current_a",
    "target_correlation",
]


def _row(seed: int, variant: str, ridge: float, source_drift: float) -> Dict[str, object]:
    result = run_experiment(
        SimulationConfig(
            seed=seed,
            scenario_preset="ideal",
            bench_samples_per_scenario=260,
            source_model_variant=variant,
            source_model_ridge=ridge,
            source_drift_fraction=source_drift,
            idealized_hardware=True,
        )
    )["summary"]
    navigation = result["navigation_prediction"]
    joint = result["mode_metrics"]["joint"]
    return {
        "seed": seed,
        "variant": variant,
        "source_drift_fraction": source_drift,
        "holdout_field_rmse_nt": result["source_holdout_field_rmse_nt"],
        "holdout_gradient_rmse_nt_per_m": result[
            "source_holdout_gradient_rmse_nt_per_m"
        ],
        "navigation_field_rmse_nt": navigation["field_vector_rmse_nt"],
        "navigation_gxx_rmse_nt_per_m": navigation["gxx_rmse_nt_per_m"],
        "residual_field_rms_nt": joint["field_residual_rms_nt"],
        "residual_gxx_rms_nt_per_m": joint["gxx_residual_rms_nt_per_m"],
        "peak_required_current_a": joint["peak_current_a"],
        "target_correlation": result["target_protection"]["waveform_correlation"],
    }


def _aggregate(rows: List[Dict[str, object]]) -> Dict[str, object]:
    output = {"case_count": len(rows)}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        output[metric] = {
            "mean": float(values.mean()),
            "max": float(values.max()),
            "p95": float(np.percentile(values, 95)),
        }
    return output


def validate_upgrade(seeds: Iterable[int]) -> Dict[str, object]:
    seeds = list(seeds)
    legacy = [_row(seed, "legacy", 0.02, 0.0) for seed in seeds]
    enhanced = [_row(seed, "enhanced_dynamic", 5.0e-7, 0.0) for seed in seeds]
    drift_rows = []
    for drift in [0.02, 0.04, 0.08]:
        drift_rows.extend(
            [_row(seed, "enhanced_dynamic", 5.0e-7, drift) for seed in seeds]
        )
    legacy_aggregate = _aggregate(legacy)
    enhanced_aggregate = _aggregate(enhanced)
    drift_aggregate = {
        str(drift): _aggregate(
            [row for row in drift_rows if row["source_drift_fraction"] == drift]
        )
        for drift in [0.02, 0.04, 0.08]
    }
    acceptance = {
        "enhanced_nominal_navigation_field_max_le_0_1_nt": enhanced_aggregate[
            "navigation_field_rmse_nt"
        ]["max"]
        <= 0.1,
        "enhanced_nominal_navigation_gxx_max_le_0_25_nt_per_m": enhanced_aggregate[
            "navigation_gxx_rmse_nt_per_m"
        ]["max"]
        <= 0.25,
        "enhanced_nominal_residual_field_max_le_0_1_nt": enhanced_aggregate[
            "residual_field_rms_nt"
        ]["max"]
        <= 0.1,
        "enhanced_nominal_residual_gxx_max_le_0_25_nt_per_m": enhanced_aggregate[
            "residual_gxx_rms_nt_per_m"
        ]["max"]
        <= 0.25,
        "target_correlation_min_ge_0_999999": min(
            row["target_correlation"] for row in enhanced
        )
        >= 0.999999,
    }
    return {
        "seeds": seeds,
        "assumptions": {
            "idealized_hardware": True,
            "current_limit": "disabled",
            "driver_lag": "disabled",
            "coil_model_mismatch": "disabled",
            "sensor_noise": "disabled",
        },
        "legacy_nominal": legacy_aggregate,
        "enhanced_nominal": enhanced_aggregate,
        "source_drift_stress": drift_aggregate,
        "acceptance": acceptance,
        "all_acceptance_passed": all(acceptance.values()),
        "cases": legacy + enhanced + drift_rows,
    }


def save_validation_report(report: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = report["cases"]
    with (output_dir / "validation_cases.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
