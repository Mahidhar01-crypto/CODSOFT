# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load Dataset
data = pd.read_csv("fraudTest.csv")

# Display first 5 rows
print(data.head())

# Dataset Information
print("\nDataset Shape:", data.shape)
print(data.info())

# Check Missing Values
print("\nMissing Values:")
print(data.isnull().sum())

# Fraud vs Normal Transactions
fraud = data[data["is_fraud"] == 1]
normal = data[data["is_fraud"] == 0]

print("\nFraud Cases:", len(fraud))
print("Normal Cases:", len(normal))

# Plot Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="is_fraud", data=data)
plt.title("Fraud vs Normal Transactions")
plt.show()

# -----------------------------
# Data Preprocessing
# -----------------------------

# Drop unnecessary columns
drop_cols = [
    'Unnamed: 0',
    'trans_date_trans_time',
    'first',
    'last',
    'street',
    'city',
    'state',
    'job',
    'dob',
    'trans_num'
]

data = data.drop(columns=drop_cols)

# Convert categorical columns to numeric
data = pd.get_dummies(data, drop_first=True)

# Features and Target
X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Random Forest Model
# -----------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Fraud"],
    yticklabels=["Normal", "Fraud"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
