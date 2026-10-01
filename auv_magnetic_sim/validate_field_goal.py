import argparse
import json
from datetime import datetime
from pathlib import Path

from magclean_sim.field_validation import (
    save_field_goal_validation,
    validate_field_goal,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the <1 nT field-residual goal under high maneuver"
    )
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    seed_count = max(3, args.seeds)
    report = validate_field_goal(range(20260923, 20260923 + seed_count))
    output = Path(__file__).resolve().parent / "runs" / (
        "field_goal_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    save_field_goal_validation(report, output)
    print(
        json.dumps(
            {
                "aggregate": report["aggregate"],
                "criteria": report["criteria"],
                "all_criteria_passed": report["all_criteria_passed"],
                "ablations": report["ablations"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Saved:", output)


if __name__ == "__main__":
    main()
