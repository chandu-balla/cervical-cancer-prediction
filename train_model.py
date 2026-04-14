import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================
# Load dataset
# ==============================
DATA_PATH = r"C:\Users\chand\Downloads\risk_factors_cervical_cancer.csv"
data = pd.read_csv(DATA_PATH)

# ==============================
# Data cleaning
# ==============================
data.replace("?", np.nan, inplace=True)

for col in data.columns:
    if data[col].dtype == "object":
        data[col] = pd.to_numeric(data[col], errors="coerce")

# ==============================
# Feature engineering
# ==============================
data["Sexual_activity_years"] = data["Age"] - data["First sexual intercourse"]
data["Smoker_flag"] = (data["Smokes"] > 0).astype(int)
data["Hormonal_years_flag"] = (
    data["Hormonal Contraceptives (years)"] > 5
).astype(int)
data["Screening_positive_count"] = (
    data["Hinselmann"] +
    data["Schiller"] +
    data["Citology"]
)

# ==============================
# FINAL feature selection (IMPORTANT)
# ==============================
final_features = [
    "Age",
    "Number of sexual partners",
    "Num of pregnancies",
    "Sexual_activity_years",
    "Smoker_flag",
    "Hormonal_years_flag",
    "Screening_positive_count",
    "Dx:HPV",
    "Biopsy"
]

data = data[final_features]

# ==============================
# Handle missing values
# ==============================
data.fillna(data.median(), inplace=True)

# ==============================
# Split X & y
# ==============================
X = data.drop("Biopsy", axis=1)
y = data["Biopsy"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# Scaling (ONLY 8 FEATURES)
# ==============================
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# ==============================
# Model
# ==============================
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear"
)
model.fit(x_train_scaled, y_train)

# ==============================
# Evaluation
# ==============================
y_pred = model.predict(x_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# Save model & scaler
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(BASE_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ model.pkl and scaler.pkl saved (8-feature compatible)")