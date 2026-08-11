from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["preprocessor_loaded"] is True


def test_prediction():
    payload = {
        "total_investment": 50000,
        "current_balance": 12000,
        "marital_status": "Married",
        "gender": "Male",
        "due_payment": 2500,
        "compensation_charged": "No",
        "client_type": "Individual",
        "repay_mode": "Monthly",
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in {"Good", "Bad"}

    assert 0 <= result["probability_good"] <= 1
    assert 0 <= result["probability_bad"] <= 1

    total_probability = result["probability_good"] + result["probability_bad"]

    assert abs(total_probability - 1.0) < 0.01


def test_invalid_request():
    response = client.post(
        "/predict",
        json={
            "total_investment": 50000,
        },
    )

    assert response.status_code == 422
