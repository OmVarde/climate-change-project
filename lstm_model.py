import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle

# Load data
data = pd.read_csv("weatherAUS.csv")
data = data.dropna()

# Use rainfall only for time-series
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date")

rainfall = data["Rainfall"].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
rainfall_scaled = scaler.fit_transform(rainfall)

# Create sequences
X = []
y = []

time_step = 30

for i in range(time_step, len(rainfall_scaled)):
    X.append(rainfall_scaled[i-time_step:i])
    y.append(rainfall_scaled[i])

X, y = np.array(X), np.array(y)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train
model.fit(X, y, epochs=5, batch_size=32)

# Save model
model.save("lstm_model.h5")

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ LSTM model trained and saved")