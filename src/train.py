from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
PARAMS_PATH = Path("params.yaml")


def load_parameters() -> dict:
    """Load training parameters from params.yaml."""
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        parameters = yaml.safe_load(file)

    if "train" not in parameters:
        raise KeyError("The 'train' section is missing from params.yaml.")

    return parameters["train"]


def load_data():
    """Load processed training and testing datasets."""
    required_files = [
        PROCESSED_DIR / "X_train.csv",
        PROCESSED_DIR / "X_test.csv",
        PROCESSED_DIR / "y_train.csv",
        PROCESSED_DIR / "y_test.csv",
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing file: {file_path}. Run prepare.py first."
            )

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(
        PROCESSED_DIR / "y_train.csv"
    ).squeeze("columns")
    y_test = pd.read_csv(
        PROCESSED_DIR / "y_test.csv"
    ).squeeze("columns")

    return X_train, X_test, y_train, y_test


def create_model(parameters: dict):
    """Create the model specified in params.yaml."""
    model_type = parameters["model_type"]

    if model_type == "logistic_regression":
        return LogisticRegression(
            C=float(parameters.get("C", 1.0)),
            max_iter=int(parameters.get("max_iter", 1000)),
            random_state=int(parameters.get("random_state", 42)),
            class_weight=parameters.get("class_weight"),
        )

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(
                parameters.get("n_estimators", 100)
            ),
            max_depth=parameters.get("max_depth"),
            min_samples_split=int(
                parameters.get("min_samples_split", 2)
            ),
            random_state=int(parameters.get("random_state", 42)),
            class_weight=parameters.get("class_weight"),
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def calculate_metrics(model, X_test, y_test) -> dict:
    """Calculate classification metrics."""
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_bad": precision_score(
            y_test,
            predictions,
            pos_label="Bad",
            zero_division=0,
        ),
        "recall_bad": recall_score(
            y_test,
            predictions,
            pos_label="Bad",
            zero_division=0,
        ),
        "f1_bad": f1_score(
            y_test,
            predictions,
            pos_label="Bad",
            zero_division=0,
        ),
    }

    if hasattr(model, "predict_proba"):
        bad_class_index = list(model.classes_).index("Bad")
        bad_probabilities = model.predict_proba(X_test)[
            :, bad_class_index
        ]

        binary_target = (y_test == "Bad").astype(int)

        metrics["roc_auc"] = roc_auc_score(
            binary_target,
            bad_probabilities,
        )

    return metrics


def main() -> None:
    parameters = load_parameters()
    X_train, X_test, y_train, y_test = load_data()

    model = create_model(parameters)

    experiment_name = parameters.get(
        "experiment_name",
        "loan-default-prediction",
    )
    run_name = parameters.get(
        "run_name",
        parameters["model_type"],
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)

        metrics = calculate_metrics(model, X_test, y_test)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "model.joblib"
        joblib.dump(model, model_path)

        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_path))
        mlflow.sklearn.log_model(model, name="model")

        print("\nTraining completed successfully.")
        print(f"Run name: {run_name}")
        print(f"Model type: {parameters['model_type']}")

        print("\nMetrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()