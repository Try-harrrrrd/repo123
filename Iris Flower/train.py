import os
import json
import pickle
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# LOAD IRIS DATASET
# ==========================================

iris = load_iris(as_frame=True)

X = iris.data
y = iris.target

df = pd.concat([X, pd.Series(y, name="target")], axis=1)

print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeature names:")
print(iris.feature_names)

print("\nTarget names:")
print(iris.target_names.tolist())

print("\nFirst 5 rows:")
print(df.head())

# ==========================================
# VISUALIZE DATASET
# ==========================================

pd.options.display.max_columns = None

scatter_matrix(
    X,
    figsize=(10, 10),
    diagonal="hist"
)

plt.suptitle(
    "Iris Feature Pairwise Scatter Matrix",
    y=0.92
)

plt.show()

# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# TRAIN MODEL
# ==========================================

rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_classifier.fit(X_train, y_train)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nRandomForest Test Accuracy:")
print(f"{accuracy:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:\n")
print(cm)

plt.figure(figsize=(4, 3))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title("Confusion Matrix")

plt.colorbar()

plt.xticks(
    range(len(iris.target_names)),
    iris.target_names,
    rotation=45
)

plt.yticks(
    range(len(iris.target_names)),
    iris.target_names
)

plt.xlabel("Predicted")

plt.ylabel("True")

plt.tight_layout()

plt.show()

# ==========================================
# CREATE MODELS FOLDER
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

# ==========================================
# SAVE MODEL USING JOBLIB
# ==========================================

joblib.dump(
    rf_classifier,
    "models/iris_model.joblib"
)

# ==========================================
# SAVE MODEL USING PICKLE
# ==========================================

with open(
    "models/iris_model.pickle",
    "wb"
) as f:
    pickle.dump(
        rf_classifier,
        f
    )

# ==========================================
# SAVE MODEL INFORMATION
# ==========================================

model_info = {
    "model_type": "RandomForestClassifier",
    "accuracy": float(accuracy),
    "feature_names": iris.feature_names,
    "target_names": iris.target_names.tolist()
}

with open(
    "models/model_info.json",
    "w"
) as f:
    json.dump(
        model_info,
        f,
        indent=4
    )

# ==========================================
# FEATURE RANGES FOR STREAMLIT SLIDERS
# ==========================================

feature_ranges = {

    "sepal_length": {
        "min": float(
            X["sepal length (cm)"].min()
        ),
        "max": float(
            X["sepal length (cm)"].max()
        ),
        "default": float(
            X["sepal length (cm)"].mean()
        )
    },

    "sepal_width": {
        "min": float(
            X["sepal width (cm)"].min()
        ),
        "max": float(
            X["sepal width (cm)"].max()
        ),
        "default": float(
            X["sepal width (cm)"].mean()
        )
    },

    "petal_length": {
        "min": float(
            X["petal length (cm)"].min()
        ),
        "max": float(
            X["petal length (cm)"].max()
        ),
        "default": float(
            X["petal length (cm)"].mean()
        )
    },

    "petal_width": {
        "min": float(
            X["petal width (cm)"].min()
        ),
        "max": float(
            X["petal width (cm)"].max()
        ),
        "default": float(
            X["petal width (cm)"].mean()
        )
    }
}

with open(
    "models/feature_ranges.json",
    "w"
) as f:
    json.dump(
        feature_ranges,
        f,
        indent=4
    )

# ==========================================
# SUCCESS MESSAGE
# ==========================================

print("\nSaved successfully!")

print("models/iris_model.joblib")
print("models/iris_model.pickle")
print("models/model_info.json")
print("models/feature_ranges.json")