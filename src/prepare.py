import os
import joblib
import yaml
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------
# Load parameters
# -----------------------

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

test_size = params["prepare"]["test_size"]
random_state = params["prepare"]["random_state"]


# -----------------------
# Read Dataset
# -----------------------

df = pd.read_csv("data/raw/credit_risk_dataset.csv")


# -----------------------
# Remove Duplicates
# -----------------------

df = df.drop_duplicates()


# -----------------------
# Split Features & Target
# -----------------------

X = df.drop("loan_quality", axis=1)
y = df["loan_quality"]


# -----------------------
# Identify Columns
# -----------------------

numerical_features = X.select_dtypes(include=["int64", "float64"]).columns

categorical_features = X.select_dtypes(include=["object"]).columns


# -----------------------
# Preprocessing
# -----------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# -----------------------
# Train/Test Split
# -----------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=y
)


# -----------------------
# Transform Data
# -----------------------

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


# -----------------------
# Save Files
# -----------------------

os.makedirs("data/processed", exist_ok=True)

pd.DataFrame(X_train).to_csv(
    "data/processed/X_train.csv",
    index=False
)

pd.DataFrame(X_test).to_csv(
    "data/processed/X_test.csv",
    index=False
)

y_train.to_csv(
    "data/processed/y_train.csv",
    index=False
)

y_test.to_csv(
    "data/processed/y_test.csv",
    index=False
)

joblib.dump(
    preprocessor,
    "data/processed/preprocessor.pkl"
)

print("Data preparation completed successfully!")