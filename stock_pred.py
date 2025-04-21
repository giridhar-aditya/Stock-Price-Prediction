import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.linear_model import LinearRegression

# Load dataset from uploaded CSV file
df = pd.read_csv(r"D:\\ML Projects\\Stock Price Prediction\\stock_data.csv", usecols=["Date", "Close"])

df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

# Prepare data for LSTM
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 50
X, y = create_sequences(data_scaled)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Predict with LSTM
y_pred_lstm = model.predict(X_test)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Train Linear Regression as baseline
X_lr = np.arange(len(df)).reshape(-1, 1)
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, df["Close"], test_size=0.2, random_state=42, shuffle=False)
reg = LinearRegression()
reg.fit(X_train_lr, y_train_lr)
y_pred_lr = reg.predict(X_test_lr)

# Evaluate models
mae_lstm = mean_absolute_error(y_test_actual, y_pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test_actual, y_pred_lstm))
r2_lstm = r2_score(y_test_actual, y_pred_lstm)

mae_lr = mean_absolute_error(y_test_lr, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test_lr, y_pred_lr))
r2_lr = r2_score(y_test_lr, y_pred_lr)

print(f"LSTM - MAE: {mae_lstm:.4f}, RMSE: {rmse_lstm:.4f}, R2: {r2_lstm:.4f}")
print(f"Regression - MAE: {mae_lr:.4f}, RMSE: {rmse_lr:.4f}, R2: {r2_lr:.4f}")

# Plot results
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label='Actual Prices')
plt.plot(y_pred_lstm, label='LSTM Predictions')
plt.plot(y_pred_lr, label='Regression Predictions')
plt.legend()
plt.title('Stock Price Prediction')
plt.show()
