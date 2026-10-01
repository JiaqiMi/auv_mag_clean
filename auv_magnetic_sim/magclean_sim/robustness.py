import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .config import SimulationConfig
from .experiment import run_experiment


def run_robustness_matrix(
    base: SimulationConfig,
    seeds: Iterable[int],
    source_drifts: Iterable[float],
    coil_drifts: Iterable[float],
) -> Dict[str, object]:
    """Run a deterministic factor matrix and report screening pass rates."""
    rows: List[Dict[str, float]] = []
    for seed in seeds:
        for source_drift in source_drifts:
            for coil_drift in coil_drifts:
                values = base.to_dict()
                values.update(
                    {
                        "seed": int(seed),
                        "source_drift_fraction": float(source_drift),
                        "coil_drift_fraction": float(coil_drift),
                        "persist_run": False,
                        "max_plot_points": 80,
                    }
                )
                result = run_experiment(SimulationConfig.from_dict(values))
                joint = result["summary"]["mode_metrics"]["joint"]
                target = result["summary"]["target_protection"]
                row = {
                    "seed": int(seed),
                    "source_drift_fraction": float(source_drift),
                    "coil_drift_fraction": float(coil_drift),
                    "field_reduction_percent": joint["field_reduction_percent"],
                    "gxx_reduction_percent": joint["gxx_reduction_percent"],
                    "saturation_fraction": joint["saturation_fraction"],
                    "target_transfer_ratio": target["transfer_ratio"],
                    "target_waveform_correlation": target["waveform_correlation"],
                }
                row["screening_pass"] = bool(
                    row["field_reduction_percent"] >= 60.0
                    and row["gxx_reduction_percent"] >= 25.0
                    and row["saturation_fraction"] <= 0.30
                    and 0.90 <= row["target_transfer_ratio"] <= 1.10
                    and row["target_waveform_correlation"] >= 0.98
                )
                rows.append(row)
    numeric_names = [
        "field_reduction_percent",
        "gxx_reduction_percent",
        "saturation_fraction",
        "target_transfer_ratio",
        "target_waveform_correlation",
    ]
    aggregate = {}
    for name in numeric_names:
        values = np.asarray([row[name] for row in rows])
        aggregate[name] = {
            "mean": float(values.mean()),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
            "worst": float(values.min() if "reduction" in name or "correlation" in name else values.max()),
        }
    aggregate["screening_pass_rate"] = float(np.mean([row["screening_pass"] for row in rows]))
    return {"case_count": len(rows), "aggregate": aggregate, "cases": rows}


def save_robustness_report(report: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "robustness_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = report["cases"]
    with (output_dir / "robustness_cases.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
