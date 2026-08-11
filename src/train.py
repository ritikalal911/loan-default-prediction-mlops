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
        parameters = yaml.safe_load(file) or {}

    if "train" not in parameters:
        raise KeyError(
            "The 'train' section is missing from params.yaml."
        )

    return parameters["train"]


def load_data():
    """
    Load training and validation datasets.

    The test set is intentionally not loaded here.
    It is reserved for final evaluation in evaluate.py.
    """

    required_files = [
        PROCESSED_DIR / "X_train.csv",
        PROCESSED_DIR / "X_val.csv",
        PROCESSED_DIR / "y_train.csv",
        PROCESSED_DIR / "y_val.csv",
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing file: {file_path}. "
                "Run src/prepare.py first."
            )

    X_train = pd.read_csv(
        PROCESSED_DIR / "X_train.csv"
    )

    X_val = pd.read_csv(
        PROCESSED_DIR / "X_val.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DIR / "y_train.csv"
    ).squeeze("columns")

    y_val = pd.read_csv(
        PROCESSED_DIR / "y_val.csv"
    ).squeeze("columns")

    return X_train, X_val, y_train, y_val


def create_model(parameters: dict):
    """Create the model specified in params.yaml."""

    model_type = parameters["model_type"]

    if model_type == "logistic_regression":
        return LogisticRegression(
            C=float(parameters.get("C", 1.0)),
            max_iter=int(
                parameters.get("max_iter", 1000)
            ),
            random_state=int(
                parameters.get("random_state", 42)
            ),
            class_weight=parameters.get(
                "class_weight"
            ),
        )

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(
                parameters.get(
                    "n_estimators",
                    100,
                )
            ),
            max_depth=parameters.get(
                "max_depth"
            ),
            min_samples_split=int(
                parameters.get(
                    "min_samples_split",
                    2,
                )
            ),
            random_state=int(
                parameters.get(
                    "random_state",
                    42,
                )
            ),
            class_weight=parameters.get(
                "class_weight"
            ),
        )

    raise ValueError(
        f"Unsupported model type: {model_type}"
    )


def calculate_metrics(
    model,
    X_val,
    y_val,
) -> dict:
    """Calculate validation metrics."""

    predictions = model.predict(X_val)

    metrics = {
        "validation_accuracy": float(
            accuracy_score(
                y_val,
                predictions,
            )
        ),
        "validation_precision_bad": float(
            precision_score(
                y_val,
                predictions,
                pos_label="Bad",
                zero_division=0,
            )
        ),
        "validation_recall_bad": float(
            recall_score(
                y_val,
                predictions,
                pos_label="Bad",
                zero_division=0,
            )
        ),
        "validation_f1_bad": float(
            f1_score(
                y_val,
                predictions,
                pos_label="Bad",
                zero_division=0,
            )
        ),
    }

    if hasattr(model, "predict_proba"):

        if "Bad" not in model.classes_:
            raise ValueError(
                "Expected target class 'Bad' "
                "in model.classes_."
            )

        bad_class_index = list(
            model.classes_
        ).index("Bad")

        bad_probabilities = model.predict_proba(
            X_val
        )[:, bad_class_index]

        binary_target = (
            y_val == "Bad"
        ).astype(int)

        metrics["validation_roc_auc"] = float(
            roc_auc_score(
                binary_target,
                bad_probabilities,
            )
        )

    return metrics


def main() -> None:

    parameters = load_parameters()

    (
        X_train,
        X_val,
        y_train,
        y_val,
    ) = load_data()

    model = create_model(parameters)

    experiment_name = parameters.get(
        "experiment_name",
        "loan-default-prediction",
    )

    run_name = parameters.get(
        "run_name",
        parameters["model_type"],
    )

    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    mlflow.set_experiment(
        experiment_name
    )

    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    print(
        f"\nExperiment: {experiment_name}"
    )

    print(
        f"Run name: {run_name}"
    )

    print(
        f"Model type: "
        f"{parameters['model_type']}"
    )

    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Validation samples: "
        f"{len(X_val)}"
    )

    with mlflow.start_run(
        run_name=run_name
    ):

        # Train only using training data
        model.fit(
            X_train,
            y_train,
        )

        # Evaluate experiment using
        # validation data
        metrics = calculate_metrics(
            model,
            X_val,
            y_val,
        )

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = (
            MODEL_DIR / "model.joblib"
        )

        joblib.dump(
            model,
            model_path,
        )

        # Log model configuration
        mlflow.log_params(
            parameters
        )

        # Log validation metrics
        mlflow.log_metrics(
            metrics
        )

        # Save model artifact
        mlflow.log_artifact(
            str(model_path)
        )

        mlflow.sklearn.log_model(
            model,
            name="model",
        )

        print(
            "\nTraining completed "
            "successfully."
        )

        print(
            f"Model saved to: "
            f"{model_path}"
        )

        print(
            "\nValidation metrics:"
        )

        for (
            metric_name,
            metric_value,
        ) in metrics.items():

            print(
                f"{metric_name}: "
                f"{metric_value:.4f}"
            )


if __name__ == "__main__":
    main()