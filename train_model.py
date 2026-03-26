import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("weatherAUS.csv")
data.columns = data.columns.str.strip()

# Drop missing values
data = data.dropna()

# Features (aligned with UI)
features = [
    "MinTemp", "MaxTemp", "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm"
]

X = data[features]
y = data["RainTomorrow"]

# Convert target
y = y.map({"Yes": 1, "No": 0})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved")