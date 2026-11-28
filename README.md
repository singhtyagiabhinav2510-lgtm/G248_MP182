# G248 - MP182 Bakery Bread Sales Prediction

## Project Overview
This project is part of the SCSE3040 MLOps Mini Project.
It focuses on predicting daily bread sales for bakery outlets
in Jaipur using machine learning and an MLOps pipeline.

## Problem Statement
Predict the number of bread loaves sold per day at different
bakery outlets to reduce wastage and prevent stock shortages.

## Group Details
Group ID: G248
Problem ID: MP182
Batch: 57
Faculty: Dr. Upasna Joshi

## Group Members
S24CSEU1629 - Ishika Singh
S24CSEU1019 - Aditya Aggarwal
S24CSEU0196 - Abhinav Tyagi
S24CSEU0629 - Ananya Chatterjee

## Objective
Build an end-to-end machine learning pipeline that covers data
versioning, data preprocessing, model training, experiment tracking,
model deployment, monitoring, and retraining.

## Technology Stack
Python
Git and GitHub
DVC
Pandas
Scikit-learn
MLflow
FastAPI
Docker
GitHub Actions
Kubernetes

## Stage 4 — Monitoring and Retraining

Run the Stage 4 workflow with Python 3.12:

```bash
python3.12 -m src.bakery_sales.monitor
```

The workflow cleans training and later data using the existing preprocessing convention, calculates 10-bin PSI for the six numerical features, evaluates v1 separately for College Gate, Mall Road, and Station Road, sorts `later.csv` chronologically, retrains Ridge v2 on cleaned train data plus the first 210 later rows, and evaluates v1 and v2 on the final 90 rows.

V2 is promoted to the `champion` MLflow alias only when `MAE_v2 <= 0.85 * MAE_v1`. The workflow predicts all 250 holdout records and writes `predictions/holdout_predictions.csv` with exactly `record_id,prediction` columns.

Run all tests with:

```bash
python3.12 -m pytest -v
```