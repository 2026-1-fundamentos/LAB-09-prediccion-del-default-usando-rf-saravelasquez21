# flake8: noqa: E501


import gzip
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


INPUT_DIR = Path("files/input")
MODEL_FILE = Path("files/models/model.pkl.gz")
METRICS_FILE = Path("files/output/metrics.json")


def clean_data(dataframe):
    """Clean one input dataframe according to the homework statement."""
    dataframe = dataframe.copy()
    dataframe = dataframe.rename(columns={"default payment next month": "default"})
    dataframe = dataframe.drop(columns=["ID"])
    dataframe = dataframe.dropna()

    dataframe.loc[dataframe["EDUCATION"] > 4, "EDUCATION"] = 4
    dataframe = dataframe[dataframe["EDUCATION"] != 0]
    dataframe = dataframe[dataframe["MARRIAGE"] != 0]

    return dataframe


def split_features_target(train_data, test_data):
    """Split cleaned datasets into features and target."""
    x_train = train_data.drop(columns=["default"])
    y_train = train_data["default"]
    x_test = test_data.drop(columns=["default"])
    y_test = test_data["default"]

    return x_train, y_train, x_test, y_test


def build_model():
    """Create the optimized classification pipeline."""
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=42,
                    max_features="sqrt",
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [200],
        "classifier__max_depth": [None],
        "classifier__min_samples_split": [10],
        "classifier__class_weight": [None],
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=1,
    )


def model_metrics(model, x_data, y_data, dataset):
    """Compute classification metrics for one dataset."""
    y_pred = model.predict(x_data)

    return {
        "type": "metrics",
        "dataset": dataset,
        "precision": precision_score(y_data, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_data, y_pred),
        "recall": recall_score(y_data, y_pred, zero_division=0),
        "f1_score": f1_score(y_data, y_pred, zero_division=0),
    }


def confusion_matrix_metrics(model, x_data, y_data, dataset):
    """Compute confusion matrix metrics for one dataset."""
    y_pred = model.predict(x_data)
    matrix = confusion_matrix(y_data, y_pred, labels=[0, 1])

    return {
        "type": "cm_matrix",
        "dataset": dataset,
        "true_0": {
            "predicted_0": int(matrix[0, 0]),
            "predicted_1": int(matrix[0, 1]),
        },
        "true_1": {
            "predicted_0": int(matrix[1, 0]),
            "predicted_1": int(matrix[1, 1]),
        },
    }


def save_model(model):
    """Save the trained model compressed with gzip."""
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(MODEL_FILE, "wb") as file:
        pickle.dump(model, file)


def save_metrics(metrics):
    """Save metrics as one JSON dictionary per line."""
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        for metric in metrics:
            file.write(json.dumps(metric) + "\n")


def main():
    """Run the complete homework workflow."""
    train_data = pd.read_csv(INPUT_DIR / "train_data.csv.zip")
    test_data = pd.read_csv(INPUT_DIR / "test_data.csv.zip")

    train_data = clean_data(train_data)
    test_data = clean_data(test_data)

    x_train, y_train, x_test, y_test = split_features_target(train_data, test_data)

    model = build_model()
    model.fit(x_train, y_train)
    save_model(model)

    metrics = [
        model_metrics(model, x_train, y_train, "train"),
        model_metrics(model, x_test, y_test, "test"),
        confusion_matrix_metrics(model, x_train, y_train, "train"),
        confusion_matrix_metrics(model, x_test, y_test, "test"),
    ]
    save_metrics(metrics)


if __name__ == "__main__":
    main()