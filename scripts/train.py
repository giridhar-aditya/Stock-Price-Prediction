import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn, optim
import torch
from torch.utils.data import Dataset, DataLoader

# 1️⃣ Load dataset
df = pd.read_csv("aapl_dataset.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# 2️⃣ Feature selection
features = df.columns.tolist()
data = df[features].values

# 3️⃣ Scale data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 4️⃣ Prepare sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, features.index("Close")])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(data_scaled, seq_length)

# 5️⃣ Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 6️⃣ PyTorch Dataset
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = StockDataset(X_train, y_train)
test_dataset = StockDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 7️⃣ Model: CNN + BiLSTM + Transformer hybrid
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
        # x: [batch, seq_len, features]
        x = x.permute(0, 2, 1)  # for CNN: [batch, features, seq_len]
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # back to [batch, seq_len, features]
        lstm_out, _ = self.bilstm(x)
        trans_out = self.transformer(lstm_out)
        out = trans_out[:, -1, :]
        return self.fc(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SOTAStockModel(input_size=len(features)).to(device)

# 8️⃣ Loss & optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 9️⃣ Training
best_val_loss = np.inf
epochs = 50
for epoch in range(epochs):
    model.train()
    train_losses = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            val_loss = criterion(output, y_batch)
            val_losses.append(val_loss.item())
    
    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)
    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "model.pth")

# 🔟 Load best model
model.load_state_dict(torch.load("model.pth"))

# 11️⃣ Evaluation
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        output = model(X_batch).cpu().numpy()
        all_preds.append(output)
        all_true.append(y_batch.cpu().numpy())

y_pred = np.vstack(all_preds).flatten()
y_true = np.vstack(all_true).flatten()

# 12️⃣ Inverse scale
close_scaler = MinMaxScaler()
close_scaler.fit(df["Close"].values.reshape(-1,1))
y_test_rescaled = close_scaler.inverse_transform(y_true.reshape(-1,1))
y_pred_rescaled = close_scaler.inverse_transform(y_pred.reshape(-1,1))

# 13️⃣ Metrics
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
accuracy = (1 - (mae / np.mean(y_test_rescaled))) * 100

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"Accuracy: {accuracy:.2f}%")

# 14️⃣ Plot results
plt.figure(figsize=(12,6))
plt.plot(y_test_rescaled, label='Actual')
plt.plot(y_pred_rescaled, label='Predicted')
plt.title("AAPL Stock Prediction (SOTA PyTorch Model)")
plt.legend()
plt.show()
