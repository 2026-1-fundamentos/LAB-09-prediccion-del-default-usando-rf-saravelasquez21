# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
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
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


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
        ("num", MinMaxScaler(), numeric_cols),
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