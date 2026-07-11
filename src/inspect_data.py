import pandas as pd

df = pd.read_csv("data/raw/credit_risk_dataset.csv")

print("=" * 50)
print("Dataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns.tolist())

print("\nData Types")
print(df.dtypes)

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

print("\nTarget Distribution")
print(df["loan_quality"].value_counts())