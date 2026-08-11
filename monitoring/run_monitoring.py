import json
import subprocess
import sys
from pathlib import Path


DRIFT_STATUS_PATH = Path(
    "reports/monitoring/drift_status.json"
)


def run_script(script: str) -> None:
    """Run another Python script and fail on errors."""

    subprocess.run(
        [
            sys.executable,
            script,
        ],
        check=True,
    )


def main() -> None:
    print(
        "\nStep 1: Running drift detection..."
    )

    run_script(
        "monitoring/drift_detection.py"
    )

    if not DRIFT_STATUS_PATH.exists():
        raise FileNotFoundError(
            "Drift status file was not generated: "
            f"{DRIFT_STATUS_PATH}"
        )

    with DRIFT_STATUS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        status = json.load(file)

    drifted_columns = int(
        status["drifted_columns"]
    )

    total_columns = int(
        status["total_columns"]
    )

    drift_share = float(
        status["drift_share"]
    )

    threshold = float(
        status["threshold"]
    )

    drift_detected = bool(
        status["dataset_drift_detected"]
    )

    print(
        "\nMonitoring Decision:"
    )

    print(
        f"Drifted columns: "
        f"{drifted_columns} / "
        f"{total_columns}"
    )

    print(
        f"Detected drift share: "
        f"{drift_share:.3f}"
    )

    print(
        f"Retraining threshold: "
        f"{threshold:.2f}"
    )

    if drift_detected:
        print(
            "\nDrift threshold exceeded."
        )

        print(
            "Triggering model retraining..."
        )

        run_script(
            "monitoring/retrain.py"
        )

        print(
            "\n================================"
        )

        print(
            "Monitoring workflow completed."
        )

        print(
            "Drift detected -> "
            "Retraining triggered"
        )

        print(
            "================================"
        )

    else:
        print(
            "\nNo significant dataset drift."
        )

        print(
            "Current model retained."
        )

        print(
            "\nMonitoring workflow completed."
        )


if __name__ == "__main__":
    main()