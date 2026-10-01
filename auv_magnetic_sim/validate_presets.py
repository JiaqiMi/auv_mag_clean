import argparse
import json
from datetime import datetime
from pathlib import Path

from magclean_sim.preset_validation import save_preset_validation, validate_presets


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all UI condition presets")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    seed_count = 3 if args.quick else max(3, args.seeds)
    report = validate_presets(range(20260923, 20260923 + seed_count))
    output = Path(__file__).resolve().parent / "runs" / (
        "preset_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    save_preset_validation(report, output)
    print(
        json.dumps(
            {
                "aggregates": report["aggregates"],
                "criteria": report["criteria"],
                "all_criteria_passed": report["all_criteria_passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Saved:", output)


if __name__ == "__main__":
    main()
