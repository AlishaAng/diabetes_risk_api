import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, f1_score
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import HistGradientBoostingClassifier

import joblib
import os
import json

RANDOM_STATE = 42

# Load data
df = pd.read_csv("data/diabetes.csv")

# Treat biologically-impossible zeros as missing values
cols_with_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_missing] = df[cols_with_missing].replace(0, np.nan)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train/val/test split so we can tune a decision threshold on val only
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

# --- Pipelines ---
# Logistic Regression pipeline with optional polynomial features

logit_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler()), # feature scaling
    ('clf', LogisticRegression(max_iter=5000,  solver="saga", random_state=RANDOM_STATE))
])

logit_param_grid = {
    # try no interactions vs degree-2 interactions
    "clf__penalty": ["l1", "l2"],
    "clf__C": np.logspace(-3, 2, 6),            # 0.001 ... 100
    "clf__class_weight": [None, "balanced"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE) 
logit_gs = GridSearchCV(
    estimator=logit_pipeline,
    param_grid=logit_param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=0
)

logit_gs.fit(X_train, y_train)
print(f"[LOGIT] Best CV AUC: {logit_gs.best_score_:.4f}")
print("[LOGIT] Best params:", logit_gs.best_params_)
best_pipe = logit_gs.best_estimator_

# --- Threshold tuning on validation set (optimise Youden's J = TPR - FPR) ---
# Logistic Regression uses a threshold of 0.5 by default; we can potentially do better by tuning it.
train_proba = best_pipe.predict_proba(X_train)[:, 1]
fpr, tpr, thr = roc_curve(y_train, train_proba)

# Drop the first threshold which is inf (tpr=fpr=0)
fpr, tpr, thr = fpr[1:], tpr[1:], thr[1:]

youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_thr = thr[best_idx]
print(f"[LOGIT] Best threshold from train (Youden J): {best_thr:.3f}")

# Evaluate on test with tuned threshold
test_proba = logit_gs.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= best_thr).astype(int)
print(f"[LOGIT] Test ROC-AUC: {roc_auc_score(y_test, test_proba):.4f}")
print("[LOGIT] Classification report @ tuned threshold:\n", classification_report(y_test, test_pred, digits=4))



# ----- HistGradientBoostingClassifier (optional second model) -----

hgb_pipeline = Pipeline([
    # For safety, still impute (HGB in newer sklearn supports NaNs, but this keeps things robust)
    ('imputer', SimpleImputer(strategy="median")),
    # No scaler – trees don't need standardisation
    ('clf', HistGradientBoostingClassifier(random_state=RANDOM_STATE))
])

hgb_param_grid = {
    "clf__learning_rate": [0.05, 0.1],
    "clf__max_depth": [3, None],
    "clf__max_iter": [100, 200],
    "clf__min_samples_leaf": [10, 20],
}

hgb_gs = GridSearchCV(
    estimator=hgb_pipeline,
    param_grid=hgb_param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=0
)

hgb_gs.fit(X_train, y_train)

print(f"[HGB] Best CV AUC: {hgb_gs.best_score_:.4f}")
print("[HGB] Best params:", hgb_gs.best_params_)

best_hgb = hgb_gs.best_estimator_

# Evaluate on test
hgb_test_proba = best_hgb.predict_proba(X_test)[:, 1]
hgb_test_pred = (hgb_test_proba >= 0.5).astype(int)  # you can threshold tune too if you like

print(f"[HGB] Test ROC-AUC: {roc_auc_score(y_test, hgb_test_proba):.4f}")
print("[HGB] Classification report @ 0.5:\n",
      classification_report(y_test, hgb_test_pred, digits=4))



# Save artifacts
lr_test_auc = roc_auc_score(y_test, test_proba)
hgb_test_auc = roc_auc_score(y_test, hgb_test_proba)

if hgb_test_auc > lr_test_auc:
    final_model = best_hgb
    final_model_name = "hgb"
    final_test_auc = hgb_test_auc
else:
    final_model = best_pipe   # logistic regression
    final_model_name = "logreg"
    final_test_auc = lr_test_auc

print(f"[FINAL] Chosen model: {final_model_name} with test AUC = {final_test_auc:.4f}")

os.makedirs("train/artifacts", exist_ok=True)
joblib.dump(final_model, "train/artifacts/diabetes_model.joblib")

with open("train/artifacts/diabetes_threshold.json", "w") as f:
    json.dump({"threshold": float(best_thr)}, f)


