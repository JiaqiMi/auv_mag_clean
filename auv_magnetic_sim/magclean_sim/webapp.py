from pathlib import Path

from flask import Flask, jsonify, request

from .config import SimulationConfig
from .experiment import run_experiment
from .presets import public_presets


def create_app(repository_root: Path = None) -> Flask:
    package_dir = Path(__file__).resolve().parent
    root = Path(repository_root or package_dir.parent)
    app = Flask(
        __name__,
        static_folder=str(root / "static"),
        static_url_path="/static",
    )
    app.config["JSON_SORT_KEYS"] = False

    @app.route("/")
    def index():
        return app.send_static_file("index.html")

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "version": "0.2.0-high-maneuver"})

    @app.route("/api/defaults")
    def defaults():
        return jsonify(SimulationConfig().to_dict())

    @app.route("/api/presets")
    def presets():
        return jsonify(public_presets())

    @app.route("/api/simulate", methods=["POST"])
    def simulate():
        try:
            config = SimulationConfig.from_dict(request.get_json(silent=True) or {})
            return jsonify(run_experiment(config, root))
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            return jsonify({"error": str(exc)}), 400

    return app


# Import kept at the bottom so app startup remains lightweight.
import numpy as np  # noqa: E402
