import argparse
import json
from datetime import datetime
from pathlib import Path

from magclean_sim.validation import save_validation_report, validate_upgrade


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the enhanced source estimator")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--quick", action="store_true", help="use three seeds")
    args = parser.parse_args()
    seed_count = 3 if args.quick else max(3, args.seeds)
    report = validate_upgrade(range(20260923, 20260923 + seed_count))
    output = Path(__file__).resolve().parent / "runs" / (
        "upgrade_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    save_validation_report(report, output)
    compact = {
        "legacy_nominal": report["legacy_nominal"],
        "enhanced_nominal": report["enhanced_nominal"],
        "source_drift_stress": report["source_drift_stress"],
        "acceptance": report["acceptance"],
        "all_acceptance_passed": report["all_acceptance_passed"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()
