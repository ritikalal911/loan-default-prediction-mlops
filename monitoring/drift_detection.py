from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from evidently import DataDefinition, Dataset, Report
from evidently.metrics import DriftedColumnsCount
from evidently.presets import DataDriftPreset


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

RAW_DATA_PATH = Path("data/raw/credit_risk_dataset.csv")

REPORT_DIR = Path("reports/monitoring")

HTML_REPORT_PATH = REPORT_DIR / "drift_report.html"
JSON_REPORT_PATH = REPORT_DIR / "drift_report.json"
DRIFT_STATUS_PATH = REPORT_DIR / "drift_status.json"


# ---------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------

TARGET_COLUMN = "loan_quality"

DROP_COLUMNS = [
    "account_number",
    TARGET_COLUMN,
]

NUMERICAL_COLUMNS = [
    "total_investment",
    "current_balance",
    "due_payment",
]

CATEGORICAL_COLUMNS = [
    "marital_status",
    "gender",
    "compensation_charged",
    "client_type",
    "repay_mode",
]

DRIFT_THRESHOLD = 0.50


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------


def load_data() -> pd.DataFrame:
    """Load and validate the original loan dataset."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset was not found: {RAW_DATA_PATH}"
        )

    dataframe = pd.read_csv(RAW_DATA_PATH)

    required_columns = set(
        NUMERICAL_COLUMNS
        + CATEGORICAL_COLUMNS
        + [
            TARGET_COLUMN,
            "account_number",
        ]
    )

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return dataframe


# ---------------------------------------------------------
# Reference/current datasets
# ---------------------------------------------------------


def create_reference_and_current(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create historical reference data and a simulated
    current production batch.

    The target and identifier are excluded because
    production input monitoring focuses on model features.
    """

    features = dataframe.drop(
        columns=DROP_COLUMNS
    ).copy()

    reference_data, current_data = (
        train_test_split(
            features,
            test_size=0.30,
            random_state=42,
        )
    )

    reference_data = reference_data.reset_index(
        drop=True
    )

    current_data = current_data.reset_index(
        drop=True
    )

    return (
        reference_data,
        current_data,
    )


# ---------------------------------------------------------
# Simulate production drift
# ---------------------------------------------------------


def simulate_production_drift(
    current_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Introduce controlled synthetic drift.

    This simulated drift is used only to demonstrate
    the Phase 2 monitoring and retraining workflow.
    It does not represent real production traffic.
    """

    drifted_data = current_data.copy()

    # -----------------------------------------------------
    # Numerical distribution shifts
    # -----------------------------------------------------

    drifted_data["total_investment"] = (
        drifted_data["total_investment"]
        * 1.40
    )

    drifted_data["current_balance"] = (
        drifted_data["current_balance"]
        * 1.30
    )

    drifted_data["due_payment"] = (
        drifted_data["due_payment"]
        * 1.50
    ) + 500

    # -----------------------------------------------------
    # Categorical distribution shifts
    # -----------------------------------------------------

    for index, column in enumerate(
        CATEGORICAL_COLUMNS
    ):
        non_missing = (
            drifted_data[column]
            .dropna()
        )

        if non_missing.empty:
            continue

        dominant_category = (
            non_missing.mode().iloc[0]
        )

        sample_indexes = drifted_data.sample(
            frac=0.80,
            random_state=42 + index,
        ).index

        drifted_data.loc[
            sample_indexes,
            column,
        ] = dominant_category

    return drifted_data


# ---------------------------------------------------------
# Evidently datasets
# ---------------------------------------------------------


def create_evidently_datasets(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> tuple[Dataset, Dataset]:
    """Convert pandas DataFrames into Evidently datasets."""

    data_definition = DataDefinition(
        numerical_columns=NUMERICAL_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
    )

    reference_dataset = Dataset.from_pandas(
        reference_data,
        data_definition=data_definition,
    )

    current_dataset = Dataset.from_pandas(
        current_data,
        data_definition=data_definition,
    )

    return (
        reference_dataset,
        current_dataset,
    )


# ---------------------------------------------------------
# Extract DriftedColumnsCount result
# ---------------------------------------------------------


def find_count_and_share(
    obj: Any,
    total_columns: int,
) -> tuple[int, float] | None:
    """
    Recursively locate the count/share output produced by
    Evidently's DriftedColumnsCount metric.

    The summary report contains only this metric, making
    this extraction independent of visual report structure.
    """

    if isinstance(obj, dict):
        if (
            "count" in obj
            and "share" in obj
        ):
            count = obj["count"]
            share = obj["share"]

            if (
                isinstance(
                    count,
                    (int, float),
                )
                and isinstance(
                    share,
                    (int, float),
                )
            ):
                count_value = int(count)
                share_value = float(share)

                if (
                    0
                    <= count_value
                    <= total_columns
                    and 0.0
                    <= share_value
                    <= 1.0
                ):
                    return (
                        count_value,
                        share_value,
                    )

        for value in obj.values():
            result = find_count_and_share(
                value,
                total_columns,
            )

            if result is not None:
                return result

    elif isinstance(obj, list):
        for item in obj:
            result = find_count_and_share(
                item,
                total_columns,
            )

            if result is not None:
                return result

    return None


# ---------------------------------------------------------
# Generate reports
# ---------------------------------------------------------


def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> dict[str, Any]:
    """
    Generate the Evidently drift report and stable
    machine-readable drift status.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        reference_dataset,
        current_dataset,
    ) = create_evidently_datasets(
        reference_data,
        current_data,
    )

    # -----------------------------------------------------
    # Full visual drift report
    # -----------------------------------------------------

    visual_report = Report(
        [
            DataDriftPreset(
                drift_share=DRIFT_THRESHOLD,
            )
        ]
    )

    visual_result = visual_report.run(
        current_data=current_dataset,
        reference_data=reference_dataset,
    )

    visual_result.save_html(
        str(HTML_REPORT_PATH)
    )

    visual_result.save_json(
        str(JSON_REPORT_PATH)
    )

    # -----------------------------------------------------
    # Explicit dataset-level drift metric
    # -----------------------------------------------------

    summary_report = Report(
        [
            DriftedColumnsCount(
                drift_share=DRIFT_THRESHOLD,
            )
        ]
    )

    summary_result = summary_report.run(
        current_data=current_dataset,
        reference_data=reference_dataset,
    )

    summary_dict = summary_result.dict()

    total_columns = (
        len(NUMERICAL_COLUMNS)
        + len(CATEGORICAL_COLUMNS)
    )

    drift_summary = find_count_and_share(
        summary_dict,
        total_columns,
    )

    if drift_summary is None:
        raise ValueError(
            "Could not extract count/share from "
            "Evidently DriftedColumnsCount result."
        )

    (
        drifted_columns,
        drift_share,
    ) = drift_summary

    dataset_drift_detected = (
        drift_share >= DRIFT_THRESHOLD
    )

    # -----------------------------------------------------
    # Save stable status JSON for automation
    # -----------------------------------------------------

    status = {
        "total_columns": total_columns,
        "drifted_columns": drifted_columns,
        "drift_share": drift_share,
        "threshold": DRIFT_THRESHOLD,
        "dataset_drift_detected": (
            dataset_drift_detected
        ),
        "simulation": True,
        "description": (
            "Current data contains intentionally "
            "simulated distribution drift for the "
            "Phase 2 monitoring demonstration."
        ),
    }

    with DRIFT_STATUS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            status,
            file,
            indent=4,
        )

    return status


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main() -> None:
    dataframe = load_data()

    (
        reference_data,
        current_data,
    ) = create_reference_and_current(
        dataframe
    )

    current_data = simulate_production_drift(
        current_data
    )

    status = generate_drift_report(
        reference_data=reference_data,
        current_data=current_data,
    )

    print(
        "\nDrift detection completed successfully."
    )

    print(
        f"Reference records: "
        f"{len(reference_data):,}"
    )

    print(
        f"Current records: "
        f"{len(current_data):,}"
    )

    print(
        "\nDrift Summary:"
    )

    print(
        f"Drifted columns: "
        f"{status['drifted_columns']} / "
        f"{status['total_columns']}"
    )

    print(
        f"Drift share: "
        f"{status['drift_share']:.3f}"
    )

    print(
        f"Drift threshold: "
        f"{status['threshold']:.2f}"
    )

    print(
        "Dataset drift detected: "
        f"{status['dataset_drift_detected']}"
    )

    print(
        f"\nHTML report saved to: "
        f"{HTML_REPORT_PATH}"
    )

    print(
        f"JSON report saved to: "
        f"{JSON_REPORT_PATH}"
    )

    print(
        f"Drift status saved to: "
        f"{DRIFT_STATUS_PATH}"
    )

    print(
        "\nNOTE: The current dataset contains "
        "intentionally simulated drift for "
        "Phase 2 demonstration."
    )


if __name__ == "__main__":
    main()