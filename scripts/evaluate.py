import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1️⃣ Load dataset
df = pd.read_csv("aapl_dataset.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# 2️⃣ Prepare data
features = df.columns.tolist()
data = df[features].values

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 3️⃣ Create sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, features.index("Close")])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(data_scaled, seq_length)

# Full dataset (only evaluation)
test_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                              torch.tensor(y, dtype=torch.float32).unsqueeze(1))
test_loader = DataLoader(test_dataset, batch_size=32)

# 4️⃣ Model definition (must match your training model)
class SOTAStockModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(SOTAStockModel, self).__init__()
        self.cnn = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bilstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_size*2, nhead=4)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
        trans_out = self.transformer(lstm_out)
        out = trans_out[:, -1, :]
        return self.fc(out)

# 5️⃣ Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SOTAStockModel(input_size=len(features)).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# 6️⃣ Evaluate
all_preds, all_true = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        output = model(X_batch).cpu().numpy()
        all_preds.append(output)
        all_true.append(y_batch.cpu().numpy())

y_pred = np.vstack(all_preds).flatten()
y_true = np.vstack(all_true).flatten()

# 7️⃣ Inverse scaling for Close price only
close_scaler = MinMaxScaler()
close_scaler.fit(df["Close"].values.reshape(-1, 1))
y_true_rescaled = close_scaler.inverse_transform(y_true.reshape(-1, 1))
y_pred_rescaled = close_scaler.inverse_transform(y_pred.reshape(-1, 1))

# 8️⃣ Metrics
mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
rmse = np.sqrt(mean_squared_error(y_true_rescaled, y_pred_rescaled))
r2 = r2_score(y_true_rescaled, y_pred_rescaled)
accuracy = (1 - (mae / np.mean(y_true_rescaled))) * 100

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"Accuracy: {accuracy:.2f}%")

# 9️⃣ Plot
plt.figure(figsize=(12,6))
plt.plot(y_true_rescaled, label='Actual')
plt.plot(y_pred_rescaled, label='Predicted')
plt.title("AAPL Stock Evaluation (Loaded model.pth)")
plt.legend()
plt.show()
