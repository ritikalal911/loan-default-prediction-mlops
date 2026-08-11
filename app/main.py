from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import LoanApplication, PredictionResponse

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "data" / "processed" / "preprocessor.pkl"


# ---------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not PREPROCESSOR_PATH.exists():
    raise FileNotFoundError(f"Preprocessor file not found: {PREPROCESSOR_PATH}")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Loan Default Prediction API",
    description=("API for predicting whether a loan is classified as Good or Bad."),
    version="1.0.0",
)


# ---------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------


@app.get("/")
def root():
    return {
        "message": "Loan Default Prediction API",
        "status": "running",
        "docs": "/docs",
    }


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "preprocessor_loaded": True,
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(application: LoanApplication):
    try:
        # Convert API request to DataFrame
        input_data = pd.DataFrame([application.model_dump()])

        # Apply the exact preprocessing fitted in Phase 1
        transformed_data = preprocessor.transform(input_data)

        # The model was trained using named processed features.
        # Re-create those column names when available.
        if hasattr(model, "feature_names_in_"):
            transformed_data = pd.DataFrame(
                transformed_data,
                columns=model.feature_names_in_,
            )

        # Make prediction
        prediction = model.predict(transformed_data)[0]

        # Get class probabilities
        probabilities = model.predict_proba(transformed_data)[0]

        classes = list(model.classes_)

        good_index = classes.index("Good")
        bad_index = classes.index("Bad")

        probability_good = float(probabilities[good_index])
        probability_bad = float(probabilities[bad_index])

        return PredictionResponse(
            prediction=str(prediction),
            probability_good=round(
                probability_good,
                4,
            ),
            probability_bad=round(
                probability_bad,
                4,
            ),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error!s}",
        ) from error
