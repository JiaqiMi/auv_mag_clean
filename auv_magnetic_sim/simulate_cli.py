import argparse
import json
from pathlib import Path

from magclean_sim.config import SimulationConfig
from magclean_sim.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AUV magnetic simulation without the UI")
    parser.add_argument("--config", help="JSON configuration file")
    parser.add_argument("--save", action="store_true", help="persist CSV/JSON artifacts under runs/")
    args = parser.parse_args()
    values = {}
    if args.config:
        values = json.loads(Path(args.config).read_text(encoding="utf-8"))
    values["persist_run"] = bool(args.save or values.get("persist_run", False))
    result = run_experiment(SimulationConfig.from_dict(values))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    if result["run_directory"]:
        print("Saved:", result["run_directory"])


if __name__ == "__main__":
    main()
