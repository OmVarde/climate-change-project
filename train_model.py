# ==============================
# Climate Rain Prediction Model
# ==============================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==============================
# 1. Load Dataset
# ==============================
print("📥 Loading dataset...")

data = pd.read_csv("data.csv")

# Clean column names
data.columns = data.columns.str.strip()

print("✅ Columns:", list(data.columns))


# ==============================
# 2. Data Cleaning
# ==============================

# Drop rows with missing target
data = data.dropna(subset=["RainTomorrow"])

# Fill missing values
for col in data.columns:
    if data[col].dtype == "object":
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].median(), inplace=True)


# ==============================
# 3. Encode Categorical Data
# ==============================

le = LabelEncoder()

for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])


# ==============================
# 4. Feature & Target Split
# ==============================

target_column = "RainTomorrow"

selected_features = [
    "MinTemp", "MaxTemp", "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm"
]

X = data[selected_features]
y = data[target_column]


# ==============================
# 5. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# 6. Model Training
# ==============================

print("🚀 Training model...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)


# ==============================
# 7. Evaluation
# ==============================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {accuracy:.2f}")
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))


# ==============================
# 8. Save Model
# ==============================

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("💾 Model saved as model.pkl")