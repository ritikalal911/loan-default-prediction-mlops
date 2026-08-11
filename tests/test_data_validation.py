from pathlib import Path

import pandas as pd

SAMPLE_DATA = Path("tests/data/sample_input.csv")

EXPECTED_COLUMNS = {
    "total_investment",
    "current_balance",
    "marital_status",
    "gender",
    "due_payment",
    "compensation_charged",
    "client_type",
    "repay_mode",
}

NUMERIC_COLUMNS = [
    "total_investment",
    "current_balance",
    "due_payment",
]


def load_sample_data():
    assert SAMPLE_DATA.exists()

    return pd.read_csv(SAMPLE_DATA)


def test_required_columns():
    dataframe = load_sample_data()

    assert set(dataframe.columns) == EXPECTED_COLUMNS


def test_no_missing_values():
    dataframe = load_sample_data()

    assert dataframe.isnull().sum().sum() == 0


def test_numeric_columns():
    dataframe = load_sample_data()

    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        assert converted.notna().all()


def test_sample_has_records():
    dataframe = load_sample_data()

    assert len(dataframe) > 0
