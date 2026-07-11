
# Dataset Documentation

## Dataset Name

Credit Risk Dataset

## Source

Kaggle Credit Risk Dataset

## Machine Learning Task

Binary Classification

The objective is to predict whether a loan is of **Good** or **Bad** quality.

## Dataset Size

- Rows: 37,408
- Columns: 10

## Features

| Feature              | Type        | Description                |
| -------------------- | ----------- | -------------------------- |
| account_number       | Integer     | Unique customer identifier |
| total_investment     | Integer     | Total investment amount    |
| current_balance      | Integer     | Current account balance    |
| marital_status       | Categorical | Customer marital status    |
| gender               | Categorical | Customer gender            |
| due_payment          | Integer     | Payment due amount         |
| compensation_charged | Categorical | Compensation status        |
| client_type          | Categorical | Client category            |
| repay_mode           | Categorical | Repayment mode             |
| loan_quality         | Target      | Good / Bad loan            |

## Target Variable

loan_quality

Classes:

- Good
- Bad

## Data Quality

### Missing Values

- marital_status → 2
- gender → 2
- compensation_charged → 2
- client_type → 103

### Duplicate Rows

0

### Class Distribution

Good → 33,254

Bad → 4,154

The dataset is imbalanced because the "Good" class is much larger than the "Bad" class.

## Planned Preprocessing

- Handle missing values
- Encode categorical variables
- Scale numerical variables
- Split into training and testing datasets
