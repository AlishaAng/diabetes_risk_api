import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import os


# Load data
df = pd.read_csv("data/diabetes.csv")
print(df.head())

cols_with_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_missing] = df[cols_with_missing].replace(0, np.nan)

# Fill missing values with median
df.fillna(df.median(), inplace=True)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_proba)

print("ROC-AUC:", auc)
print(classification_report(y_test, y_pred))

# Save artifacts
os.makedirs("train/artifacts", exist_ok=True)
joblib.dump(model, "train/artifacts/model.joblib")
joblib.dump(scaler, "train/artifacts/scaler.joblib")
print("Model and scaler saved.")