from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RAW_DATA_PATH = Path("data/raw/credit_risk_dataset.csv")
PROCESSED_DIR = Path("data/processed")
PARAMS_PATH = Path("params.yaml")

TARGET_COLUMN = "Class"


def load_parameters() -> dict:
    """Load preprocessing parameters."""
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        parameters = yaml.safe_load(file) or {}

    return parameters.get(
        "prepare",
        {
            "test_size": 0.2,
            "random_state": 42,
        },
    )


def load_dataset() -> pd.DataFrame:
    """Load and validate the fraud-detection dataset."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    dataframe = pd.read_csv(RAW_DATA_PATH)

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found. "
            f"Available columns: {list(dataframe.columns)}"
        )

    return dataframe


def prepare_features(
    dataframe: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Split and standardize the dataset."""
    X = dataframe.drop(columns=[TARGET_COLUMN]).copy()

    y = dataframe[TARGET_COLUMN].map(
        {
            0: "Good",
            1: "Bad",
        }
    )

    if y.isna().any():
        raise ValueError(
            "The Class column must contain only 0 and 1."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(parameters.get("test_size", 0.2)),
        random_state=int(parameters.get("random_state", 42)),
        stratify=y,
    )

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
    )


def save_outputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scaler: StandardScaler,
) -> None:
    """Save processed datasets and fitted scaler."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(
        PROCESSED_DIR / "X_train.csv",
        index=False,
    )
    X_test.to_csv(
        PROCESSED_DIR / "X_test.csv",
        index=False,
    )
    y_train.rename("target").to_csv(
        PROCESSED_DIR / "y_train.csv",
        index=False,
    )

    y_test.rename("target").to_csv(
        PROCESSED_DIR / "y_test.csv",
        index=False,
    )

    joblib.dump(
        scaler,
        PROCESSED_DIR / "preprocessor.pkl",
    )


def main() -> None:
    parameters = load_parameters()
    dataframe = load_dataset()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
    ) = prepare_features(
        dataframe,
        parameters,
    )

    save_outputs(
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
    )

    print("Data preparation completed successfully.")
    print(f"Dataset shape: {dataframe.shape}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print("\nTarget distribution:")
    print(dataframe[TARGET_COLUMN].value_counts())


if __name__ == "__main__":
    main()