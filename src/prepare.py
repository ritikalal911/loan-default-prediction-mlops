from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
RAW_DATA_PATH = Path("data/raw/credit_risk_dataset.csv")
PROCESSED_DIR = Path("data/processed")
PARAMS_PATH = Path("params.yaml")

# Actual target column in our dataset
TARGET_COLUMN = "loan_quality"

# Identifier column - not useful for prediction
DROP_COLUMNS = ["account_number"]


# ---------------------------------------------------------
# Load parameters
# ---------------------------------------------------------
def load_parameters() -> dict:
    """Load preprocessing parameters from params.yaml."""

    default_parameters = {
        "test_size": 0.15,
        "validation_size": 0.15,
        "random_state": 42,
    }

    if not PARAMS_PATH.exists():
        return default_parameters

    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        parameters = yaml.safe_load(file) or {}

    prepare_parameters = parameters.get("prepare", {})

    return {
        "test_size": float(
            prepare_parameters.get(
                "test_size",
                default_parameters["test_size"],
            )
        ),
        "validation_size": float(
            prepare_parameters.get(
                "validation_size",
                default_parameters["validation_size"],
            )
        ),
        "random_state": int(
            prepare_parameters.get(
                "random_state",
                default_parameters["random_state"],
            )
        ),
    }


# ---------------------------------------------------------
# Load and validate dataset
# ---------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    """Load and validate the loan-default dataset."""

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

    # Remove duplicate records if any are present
    duplicate_count = dataframe.duplicated().sum()

    if duplicate_count > 0:
        print(f"Removing {duplicate_count} duplicate rows.")
        dataframe = dataframe.drop_duplicates().reset_index(drop=True)

    # Validate target values
    valid_targets = {"Good", "Bad"}
    actual_targets = set(
        dataframe[TARGET_COLUMN].dropna().unique()
    )

    if not actual_targets.issubset(valid_targets):
        raise ValueError(
            f"Unexpected values in '{TARGET_COLUMN}': "
            f"{sorted(actual_targets)}. "
            "Expected only 'Good' and 'Bad'."
        )

    return dataframe


# ---------------------------------------------------------
# Split dataset
# ---------------------------------------------------------
def split_dataset(
    dataframe: pd.DataFrame,
    parameters: dict,
):
    """Create train, validation and test splits."""

    X = dataframe.drop(
        columns=[TARGET_COLUMN] + DROP_COLUMNS,
        errors="ignore",
    ).copy()

    y = dataframe[TARGET_COLUMN].copy()

    test_size = parameters["test_size"]
    validation_size = parameters["validation_size"]
    random_state = parameters["random_state"]

    if test_size <= 0 or validation_size <= 0:
        raise ValueError(
            "test_size and validation_size must be greater than 0."
        )

    if test_size + validation_size >= 1:
        raise ValueError(
            "test_size + validation_size must be less than 1."
        )

    # -----------------------------------------------------
    # First split:
    # 85% train+validation / 15% test
    # -----------------------------------------------------
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Validation needs to be 15% of ORIGINAL dataset.
    # After removing the test set, 85% remains.
    #
    # validation ratio inside remaining data:
    # 0.15 / 0.85 = ~0.17647
    validation_relative_size = (
        validation_size / (1.0 - test_size)
    )

    # -----------------------------------------------------
    # Second split:
    # 70% train / 15% validation
    # -----------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_relative_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ---------------------------------------------------------
# Build preprocessing pipeline
# ---------------------------------------------------------
def build_preprocessor(
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """Create preprocessing pipelines for numeric and categorical data."""

    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    print("\nNumeric columns:")
    print(numeric_columns)

    print("\nCategorical columns:")
    print(categorical_columns)

    # Numeric:
    # missing values -> median
    # then scaling
    numeric_pipeline = Pipeline(
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

    # Categorical:
    # missing values -> most common category
    # then one-hot encoding
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
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
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# ---------------------------------------------------------
# Apply preprocessing
# ---------------------------------------------------------
def preprocess_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """Fit preprocessing on training data and transform all splits."""

    preprocessor = build_preprocessor(X_train)

    # IMPORTANT:
    # fit only on training data to avoid data leakage
    X_train_processed = preprocessor.fit_transform(X_train)

    X_val_processed = preprocessor.transform(X_val)

    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index,
    )

    X_val_processed = pd.DataFrame(
        X_val_processed,
        columns=feature_names,
        index=X_val.index,
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index,
    )

    return (
        X_train_processed,
        X_val_processed,
        X_test_processed,
        preprocessor,
    )


# ---------------------------------------------------------
# Save processed outputs
# ---------------------------------------------------------
def save_outputs(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
) -> None:
    """Save processed datasets and fitted preprocessing pipeline."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_train.to_csv(
        PROCESSED_DIR / "X_train.csv",
        index=False,
    )

    X_val.to_csv(
        PROCESSED_DIR / "X_val.csv",
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

    y_val.rename("target").to_csv(
        PROCESSED_DIR / "y_val.csv",
        index=False,
    )

    y_test.rename("target").to_csv(
        PROCESSED_DIR / "y_test.csv",
        index=False,
    )

    joblib.dump(
        preprocessor,
        PROCESSED_DIR / "preprocessor.pkl",
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main() -> None:
    parameters = load_parameters()

    dataframe = load_dataset()

    print("=" * 60)
    print("LOAN DEFAULT DATA PREPARATION")
    print("=" * 60)

    print(f"\nDataset shape: {dataframe.shape}")

    print("\nTarget distribution:")
    print(dataframe[TARGET_COLUMN].value_counts())

    print("\nMissing values:")
    print(dataframe.isnull().sum())

    print(
        f"\nDuplicate rows: "
        f"{dataframe.duplicated().sum()}"
    )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_dataset(
        dataframe,
        parameters,
    )

    (
        X_train_processed,
        X_val_processed,
        X_test_processed,
        preprocessor,
    ) = preprocess_data(
        X_train,
        X_val,
        X_test,
    )

    save_outputs(
        X_train_processed,
        X_val_processed,
        X_test_processed,
        y_train,
        y_val,
        y_test,
        preprocessor,
    )

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"Training rows:   "
        f"{len(X_train_processed)}"
    )

    print(
        f"Validation rows: "
        f"{len(X_val_processed)}"
    )

    print(
        f"Testing rows:    "
        f"{len(X_test_processed)}"
    )

    print(
        f"Processed features: "
        f"{X_train_processed.shape[1]}"
    )

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nValidation target distribution:")
    print(y_val.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())


if __name__ == "__main__":
    main()