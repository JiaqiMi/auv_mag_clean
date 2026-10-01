import argparse
import json
from datetime import datetime
from pathlib import Path

from magclean_sim.config import SimulationConfig
from magclean_sim.robustness import run_robustness_matrix, save_robustness_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the strict robustness factor matrix")
    parser.add_argument("--seeds", type=int, default=10, help="number of deterministic seeds")
    parser.add_argument("--quick", action="store_true", help="run a 4-case smoke matrix")
    args = parser.parse_args()
    seed_count = 2 if args.quick else max(args.seeds, 2)
    source_drifts = [0.0, 0.04] if args.quick else [0.0, 0.04, 0.08, 0.12]
    coil_drifts = [0.0] if args.quick else [0.0, 0.03, 0.06]
    report = run_robustness_matrix(
        SimulationConfig(),
        range(20260923, 20260923 + seed_count),
        source_drifts,
        coil_drifts,
    )
    output = Path(__file__).resolve().parent / "runs" / (
        "robustness_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    save_robustness_report(report, output)
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()
