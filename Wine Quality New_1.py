# ============================================================
# WINE QUALITY PREDICTION
# RANDOM FOREST & ADABOOST
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# LOAD DATASET
# ============================================================

file_path = r"D:\Personnel\Certification\AI ML PG\Projects python\Tutorials\Wine Quality\winequality-red.csv"

df = pd.read_csv(file_path, sep=";")

# Clean Column Names
import pandas as pd

df = pd.read_csv(
    r"D:\Personnel\Certification\AI ML PG\Projects python\Tutorials\Wine Quality\winequality-red.csv"
)

print(df.columns.tolist())

# Verify quality column
if "quality" not in df.columns:
    raise ValueError(
        f"'quality' column not found.\nAvailable columns:\n{df.columns.tolist()}"
    )

print("\nFirst 5 Rows:")
print(df.head())

# ============================================================
# CREATE TARGET CLASS
# ============================================================

def quality_class(score):
    if score <= 4:
        return 0        # Low
    elif score <= 6:
        return 1        # Medium
    else:
        return 2        # High

df["Quality_Class"] = df["quality"].apply(quality_class)

print("\n" + "=" * 60)
print("QUALITY CLASS DISTRIBUTION")
print("=" * 60)

print(df["Quality_Class"].value_counts())

# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(
    ["quality", "Quality_Class"],
    axis=1
)

y = df["Quality_Class"]

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)

# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST MODEL")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# ============================================================
# RANDOM FOREST CONFUSION MATRIX
# ============================================================

rf_cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    rf_cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame(
{
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}
)

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Important Features")
print(importance_df.head(10))

plt.figure(figsize=(8,5))

sns.barplot(
    data=importance_df.head(10),
    x="Importance",
    y="Feature",
    hue="Feature",
    palette="viridis",
    legend=False
)

plt.title("Feature Importance - Random Forest")

plt.tight_layout()
plt.show()

# ============================================================
# ADABOOST
# ============================================================

print("\n" + "=" * 60)
print("ADABOOST MODEL")
print("=" * 60)

ada_model = AdaBoostClassifier(
    n_estimators=200,
    learning_rate=0.8,
    random_state=42
)

ada_model.fit(X_train, y_train)

ada_pred = ada_model.predict(X_test)

ada_accuracy = accuracy_score(
    y_test,
    ada_pred
)

print(f"\nAdaBoost Accuracy: {ada_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, ada_pred))

# ============================================================
# ADABOOST CONFUSION MATRIX
# ============================================================

ada_cm = confusion_matrix(y_test, ada_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    ada_cm,
    annot=True,
    fmt="d",
    cmap="Oranges"
)

plt.title("AdaBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# ============================================================
# MODEL COMPARISON
# ============================================================

comparison_df = pd.DataFrame(
{
    "Model": ["Random Forest", "AdaBoost"],
    "Accuracy": [rf_accuracy, ada_accuracy]
}
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(comparison_df)

plt.figure(figsize=(6,4))

sns.barplot(
    data=comparison_df,
    x="Model",
    y="Accuracy",
    hue="Model",
    palette="Set2",
    legend=False
)

plt.title("Model Accuracy Comparison")

plt.tight_layout()
plt.show()

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL PROJECT SUMMARY")
print("=" * 60)

print(f"""
Random Forest Accuracy : {rf_accuracy:.4f}
AdaBoost Accuracy      : {ada_accuracy:.4f}

Key Insights:
1. Alcohol is typically the most influential feature.
2. Volatile acidity negatively impacts wine quality.
3. Sulphates positively influence quality.
4. Random Forest generally outperforms AdaBoost.
5. Wine quality can be predicted using physicochemical properties.
""")

print("\nProject Completed Successfully!")