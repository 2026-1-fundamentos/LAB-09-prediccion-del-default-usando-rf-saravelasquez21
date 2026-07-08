import gzip
import json
import os
import pickle
import zipfile

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


def load_and_clean(filename):
    with zipfile.ZipFile(filename, "r") as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f)

    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])
    df = df.dropna()
    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
    return df


train_df = load_and_clean("files/input/train_data.csv.zip")
test_df = load_and_clean("files/input/test_data.csv.zip")

x_train = train_df.drop(columns=["default"])
y_train = train_df["default"]
x_test = test_df.drop(columns=["default"])
y_test = test_df["default"]

categorical_cols = [
    "SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2",
    "PAY_3", "PAY_4", "PAY_5", "PAY_6",
]
numeric_cols = [col for col in x_train.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=0, max_samples=0.8)),
])

param_grid = {
    "classifier__n_estimators": [500, 700],
    "classifier__max_depth": [None],
    "classifier__min_samples_split": [2, 5],
}

model = GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
    verbose=1,
)

model.fit(x_train, y_train)

os.makedirs("files/models", exist_ok=True)
with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(model, f)


def compute_metrics(model, x, y, dataset_name):
    y_pred = model.predict(x)
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0),
    }


def compute_cm(model, x, y, dataset_name):
    y_pred = model.predict(x)
    cm = confusion_matrix(y, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
    }


os.makedirs("files/output", exist_ok=True)
with open("files/output/metrics.json", "w") as f:
    for metrics_data in [
        compute_metrics(model, x_train, y_train, "train"),
        compute_metrics(model, x_test, y_test, "test"),
        compute_cm(model, x_train, y_train, "train"),
        compute_cm(model, x_test, y_test, "test"),
    ]:
        f.write(json.dumps(metrics_data) + "\n")