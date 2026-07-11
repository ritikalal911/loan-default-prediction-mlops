from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)


PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/model.joblib")
PARAMS_PATH = Path("params.yaml")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"


def load_threshold() -> float:
    """Load the classification threshold from params.yaml."""
    if not PARAMS_PATH.exists():
        return 0.5

    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        parameters = yaml.safe_load(file) or {}

    return float(parameters.get("evaluate", {}).get("threshold", 0.5))


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the prepared testing features and target."""
    x_test_path = PROCESSED_DIR / "X_test.csv"
    y_test_path = PROCESSED_DIR / "y_test.csv"

    for path in (x_test_path, y_test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}. Run src/prepare.py first."
            )

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    return X_test, y_test


def load_model():
    """Load the trained model artifact."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing trained model: {MODEL_PATH}. Run src/train.py first."
        )

    return joblib.load(MODEL_PATH)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict[str, float]:
    """Evaluate the model and save metrics and figures."""
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "The trained model must support predict_proba for thresholding "
            "and ROC-AUC evaluation."
        )

    if "Bad" not in model.classes_:
        raise ValueError(
            "Expected a target class named 'Bad' in model.classes_."
        )

    bad_class_index = list(model.classes_).index("Bad")
    bad_probabilities = model.predict_proba(X_test)[:, bad_class_index]

    predictions = pd.Series(
        ["Bad" if probability >= threshold else "Good"
         for probability in bad_probabilities],
        index=y_test.index,
    )

    binary_target = (y_test == "Bad").astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
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
            roc_auc_score(binary_target, bad_probabilities)
        ),
        "threshold": float(threshold),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    confusion_display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=["Good", "Bad"],
        display_labels=["Good", "Bad"],
    )
    confusion_display.ax_.set_title("Loan Default Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    roc_display = RocCurveDisplay.from_predictions(
        binary_target,
        bad_probabilities,
        name="Loan default model",
    )
    roc_display.ax_.set_title("Loan Default ROC Curve")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return metrics


def main() -> None:
    threshold = load_threshold()
    X_test, y_test = load_test_data()
    model = load_model()

    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold,
    )

    print("\nEvaluation completed successfully.")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Figures saved to: {FIGURES_DIR}")

    print("\nMetrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()