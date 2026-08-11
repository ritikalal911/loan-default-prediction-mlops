from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/model.joblib")
RETRAINED_MODEL_PATH = Path("models/retrained_model.joblib")
PARAMS_PATH = Path("params.yaml")

REPORT_DIR = Path("reports/monitoring")
COMPARISON_PATH = REPORT_DIR / "retraining_comparison.json"


def load_data():
    """Load train, validation, and untouched test datasets."""

    required_files = [
        PROCESSED_DIR / "X_train.csv",
        PROCESSED_DIR / "X_val.csv",
        PROCESSED_DIR / "X_test.csv",
        PROCESSED_DIR / "y_train.csv",
        PROCESSED_DIR / "y_val.csv",
        PROCESSED_DIR / "y_test.csv",
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    X_train = pd.read_csv(
        PROCESSED_DIR / "X_train.csv"
    )

    X_val = pd.read_csv(
        PROCESSED_DIR / "X_val.csv"
    )

    X_test = pd.read_csv(
        PROCESSED_DIR / "X_test.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DIR / "y_train.csv"
    ).squeeze("columns")

    y_val = pd.read_csv(
        PROCESSED_DIR / "y_val.csv"
    ).squeeze("columns")

    y_test = pd.read_csv(
        PROCESSED_DIR / "y_test.csv"
    ).squeeze("columns")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


def load_parameters() -> dict:
    """Load the selected model parameters."""

    with PARAMS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        parameters = yaml.safe_load(file)

    return parameters["train"]


def calculate_metrics(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate a model using the untouched test dataset."""

    predictions = model.predict(X_test)

    bad_index = list(
        model.classes_
    ).index("Bad")

    bad_probabilities = model.predict_proba(
        X_test
    )[:, bad_index]

    binary_target = (
        y_test == "Bad"
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),
        "precision_bad": float(
            precision_score(
                y_test,
                predictions,
                pos_label="Bad",
                zero_division=0,
            )
        ),
        "recall_bad": float(
            recall_score(
                y_test,
                predictions,
                pos_label="Bad",
                zero_division=0,
            )
        ),
        "f1_bad": float(
            f1_score(
                y_test,
                predictions,
                pos_label="Bad",
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                binary_target,
                bad_probabilities,
            )
        ),
    }


def create_retrained_model(
    parameters: dict,
):
    """Create candidate retrained Random Forest."""

    return RandomForestClassifier(
        n_estimators=int(
            parameters.get(
                "n_estimators",
                200,
            )
        ),
        max_depth=parameters.get(
            "max_depth",
            15,
        ),
        min_samples_split=int(
            parameters.get(
                "min_samples_split",
                5,
            )
        ),
        random_state=int(
            parameters.get(
                "random_state",
                42,
            )
        ),
        class_weight=parameters.get(
            "class_weight",
            "balanced",
        ),
    )


def main() -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Current model not found: {MODEL_PATH}"
        )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = load_data()

    parameters = load_parameters()

    # -----------------------------------------------------
    # Current production model
    # -----------------------------------------------------

    current_model = joblib.load(
        MODEL_PATH
    )

    current_metrics = calculate_metrics(
        current_model,
        X_test,
        y_test,
    )

    # -----------------------------------------------------
    # Retraining dataset
    # -----------------------------------------------------
    # After model selection, combine training and validation
    # data while keeping the original test dataset untouched.
    # -----------------------------------------------------

    X_retrain = pd.concat(
        [
            X_train,
            X_val,
        ],
        ignore_index=True,
    )

    y_retrain = pd.concat(
        [
            y_train,
            y_val,
        ],
        ignore_index=True,
    )

    # -----------------------------------------------------
    # Train candidate model
    # -----------------------------------------------------

    retrained_model = create_retrained_model(
        parameters
    )

    retrained_model.fit(
        X_retrain,
        y_retrain,
    )

    retrained_metrics = calculate_metrics(
        retrained_model,
        X_test,
        y_test,
    )

    # -----------------------------------------------------
    # Compare current vs retrained
    # -----------------------------------------------------

    improvement = (
        retrained_metrics["roc_auc"]
        - current_metrics["roc_auc"]
    )

    should_promote = (
        retrained_metrics["roc_auc"]
        >= current_metrics["roc_auc"]
    )

    comparison = {
        "trigger": "data_drift_detected",
        "retraining_rows": int(
            len(X_retrain)
        ),
        "test_rows": int(
            len(X_test)
        ),
        "current_model": {
            "path": str(MODEL_PATH),
            "metrics": current_metrics,
        },
        "retrained_model": {
            "path": str(
                RETRAINED_MODEL_PATH
            ),
            "metrics": retrained_metrics,
        },
        "roc_auc_change": float(
            improvement
        ),
        "promotion_recommended": bool(
            should_promote
        ),
    }

    # Always save the candidate for review.
    joblib.dump(
        retrained_model,
        RETRAINED_MODEL_PATH,
    )

    with COMPARISON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            comparison,
            file,
            indent=4,
        )

    print(
        "\nRetraining completed successfully."
    )

    print(
        f"Retraining rows: {len(X_retrain):,}"
    )

    print(
        f"Test rows: {len(X_test):,}"
    )

    print(
        "\nCurrent model metrics:"
    )

    for metric, value in (
        current_metrics.items()
    ):
        print(
            f"{metric}: {value:.4f}"
        )

    print(
        "\nRetrained model metrics:"
    )

    for metric, value in (
        retrained_metrics.items()
    ):
        print(
            f"{metric}: {value:.4f}"
        )

    print(
        f"\nROC-AUC change: "
        f"{improvement:+.4f}"
    )

    print(
        "Promotion recommended: "
        f"{should_promote}"
    )

    print(
        "\nComparison saved to: "
        f"{COMPARISON_PATH}"
    )

    print(
        "Candidate model saved to: "
        f"{RETRAINED_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()