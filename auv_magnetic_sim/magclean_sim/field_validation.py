import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .config import SimulationConfig
from .experiment import run_experiment
from .presets import SIMULATION_PRESETS


def _run_case(seed: int, overrides: Dict[str, object] = None) -> Dict[str, float]:
    values = dict(SIMULATION_PRESETS["high_maneuver"]["values"])
    values.update(overrides or {})
    values["seed"] = int(seed)
    summary = run_experiment(SimulationConfig.from_dict(values))["summary"]
    joint = summary["mode_metrics"]["joint"]
    none = summary["mode_metrics"]["none"]
    return {
        "seed": int(seed),
        "field_residual_rms_nt": joint["field_residual_rms_nt"],
        "cesium_error_rms_nt": joint["cesium_error_rms_nt"],
        "joint_principal_peak_nt_per_m": joint["principal_peak_nt_per_m"],
        "uncompensated_principal_peak_nt_per_m": none["principal_peak_nt_per_m"],
        "joint_tolerance_exceed_duration_s": joint["tolerance_exceed_duration_s"],
        "peak_current_a": joint["peak_current_a"],
        "saturation_fraction": joint["saturation_fraction"],
    }


def _stats(rows: List[Dict[str, float]], key: str) -> Dict[str, float]:
    values = np.asarray([row[key] for row in rows], dtype=float)
    return {
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def validate_field_goal(seeds: Iterable[int]) -> Dict[str, object]:
    seeds = list(seeds)
    cases = [_run_case(seed) for seed in seeds]
    metric_names = [name for name in cases[0] if name != "seed"]
    aggregate = {name: _stats(cases, name) for name in metric_names}
    ablation_seed = seeds[0]
    ablations = {
        "legacy_path": _run_case(
            ablation_seed,
            {
                "preflight_alignment_enabled": False,
                "driver_preemphasis_enabled": False,
            },
        ),
        "preflight_only": _run_case(
            ablation_seed, {"driver_preemphasis_enabled": False}
        ),
        "preemphasis_only": _run_case(
            ablation_seed, {"preflight_alignment_enabled": False}
        ),
        "combined": cases[0],
    }
    criteria = {
        "all_field_residuals_below_1_nt": aggregate["field_residual_rms_nt"]["max"] < 1.0,
        "all_cesium_errors_below_1_nt": aggregate["cesium_error_rms_nt"]["max"] < 1.0,
        "uncompensated_gradient_always_exceeds_10000": aggregate[
            "uncompensated_principal_peak_nt_per_m"
        ]["min"]
        > 10000.0,
        "joint_gradient_always_below_656": aggregate[
            "joint_principal_peak_nt_per_m"
        ]["max"]
        < 656.0,
        "no_joint_gradient_limit_exposure": aggregate[
            "joint_tolerance_exceed_duration_s"
        ]["max"]
        == 0.0,
        "no_current_saturation": aggregate["saturation_fraction"]["max"] == 0.0,
    }
    return {
        "seeds": seeds,
        "case_count": len(cases),
        "aggregate": aggregate,
        "criteria": criteria,
        "all_criteria_passed": all(criteria.values()),
        "ablations": ablations,
        "cases": cases,
    }


def save_field_goal_validation(report: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "field_goal_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "field_goal_cases.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["cases"][0]))
        writer.writeheader()
        writer.writerows(report["cases"])
