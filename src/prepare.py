from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW_DATA_PATH = Path("data/raw/credit_risk_dataset.csv")
PROCESSED_DIR = Path("data/processed")
PARAMS_PATH = Path("params.yaml")

TARGET_COLUMN = "loan_quality"
IDENTIFIER_COLUMNS = ["account_number"]


def load_parameters() -> dict:
    """Load preprocessing parameters from params.yaml."""
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {PARAMS_PATH}"
        )

    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        parameters = yaml.safe_load(file)

    if "prepare" not in parameters:
        raise KeyError(
            "The 'prepare' section is missing from params.yaml"
        )

    return parameters["prepare"]


def load_dataset() -> pd.DataFrame:
    """Load and validate the raw dataset."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_PATH}"
        )

    dataframe = pd.read_csv(RAW_DATA_PATH)

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found."
        )

    return dataframe


def main() -> None:
    """Prepare the raw dataset for model training and evaluation."""
    parameters = load_parameters()
    dataframe = load_dataset()

    original_rows = len(dataframe)

    # Remove duplicate rows.
    dataframe = dataframe.drop_duplicates().copy()

    # Remove identifier columns because they do not represent
    # useful predictive information.
    columns_to_remove = [
        column
        for column in IDENTIFIER_COLUMNS
        if column in dataframe.columns
    ]

    dataframe = dataframe.drop(columns=columns_to_remove)

    # Separate features and target.
    X = dataframe.drop(columns=[TARGET_COLUMN])
    y = dataframe[TARGET_COLUMN]

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    print(f"Numerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        verbose_feature_names_out=False,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(parameters.get("test_size", 0.2)),
        random_state=int(
            parameters.get("random_state", 42)
        ),
        stratify=y,
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_dataframe = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
    )

    X_test_dataframe = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    X_train_dataframe.to_csv(
        PROCESSED_DIR / "X_train.csv",
        index=False,
    )

    X_test_dataframe.to_csv(
        PROCESSED_DIR / "X_test.csv",
        index=False,
    )

    y_train.reset_index(drop=True).to_csv(
        PROCESSED_DIR / "y_train.csv",
        index=False,
    )

    y_test.reset_index(drop=True).to_csv(
        PROCESSED_DIR / "y_test.csv",
        index=False,
    )

    joblib.dump(
        preprocessor,
        PROCESSED_DIR / "preprocessor.joblib",
    )

    removed_duplicates = original_rows - len(dataframe)

    print("\nData preparation completed successfully.")
    print(f"Duplicate rows removed: {removed_duplicates}")
    print(f"Training rows: {len(X_train_dataframe)}")
    print(f"Testing rows: {len(X_test_dataframe)}")
    print(f"Processed features: {len(feature_names)}")
    print(f"Target classes: {sorted(y.unique().tolist())}")


if __name__ == "__main__":
    main()