# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load Dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Display first rows
print(df.head())

# Drop unwanted columns
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])

# Rename columns
df.columns = ['label', 'message']

# Convert target variable
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print(df.head())

# Dataset Information
print("\nDataset Shape:", df.shape)

# Check Missing Values
print(df.isnull().sum())

# Class Distribution
print(df['label'].value_counts())

# Visualization
plt.figure(figsize=(6,4))
sns.countplot(x=df['label'])
plt.title("Ham vs Spam Messages")
plt.show()

# Features and Target
X = df['message']
y = df['label']

# Text Vectorization
vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Ham', 'Spam'],
    yticklabels=['Ham', 'Spam']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
