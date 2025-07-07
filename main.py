import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
import os  # Added for file checking

TICKER     = "TSLA"   
LOOKBACK   = 30        # days of history to consider
TEST_SPLIT = 0.2        # 20% of the data will be used for testing
EPOCHS     = 100       # read the dataset how many x times
BATCH_SIZE = 32        # how many samples to read at once go before the next finsing the data set and then the next epoch will start    
LR         = 1e-4    # learning rate the smaller the number more accurate in a sense but u have to wait longer
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu" # where to run the model

#get the data 
df = yf.download(TICKER, period="2y", interval="1d")

# makingsure  Close is 1-dimensional for all calculations
close_series = df["Close"].squeeze()
#squeeze converts a DataFrame with a single column into a Series basically means conveting a 2d array to a 1d array removes unnecessary "wrapper" dimensions
df["Return"]     = close_series.pct_change()
df["MA5"]        = close_series.rolling(5).mean()
df["MA10"]       = close_series.rolling(10).mean()
df["Volatility"] = close_series.rolling(5).std()

# RSI relative strength index 
rsi = RSIIndicator(close=close_series, window=14)   
df["RSI"] = rsi.rsi()

# MACD moving average convergence divergence needs squeeze()
df["MACD"] = ta.trend.macd(close_series)

# Bollinger Bands needs squeeze()
#Upper Band → Middle Band + 2 standard deviations and if price is above the upper band it is considered overbought hence its expected to go down(pullback)
#Lower Band → Middle Band - 2 standard deviations and if price is below the lower band it is considered oversold hence its expected to go up(rebound)
# middle band id the 20-day moving average
bb = ta.volatility.BollingerBands(close_series, window=20)
df["BB_upper"] = bb.bollinger_hband()
df["BB_lower"] = bb.bollinger_lband()

df.dropna(inplace=True)

#  3. LABEL 
df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
#shift (-1) shifts the values in the "Close" column up by one row, so that the next day's closing price is compared to the current day's closing price
#.where converts the boolean values into 1s and 0s, where 1 indicates an upward movement (next day's price is higher) and 0 indicates a downward movement (next day's price is lower)

print("=" * 50)

print("DATA BALANCE CHECK:")
print("UP days:", (df["Target"] == 1).sum())
print("DOWN days:", (df["Target"] == 0).sum())
print("UP percentage:", (df["Target"] == 1).mean() * 100, "%")

print("=" * 50)

# 4 FEATURES & SCALING 

# #volaitility is the standard deviation of the closing price over the last 5 days
#ma is just the moving average of the closing price over the last 5 and 10 days 
#return is the percentage change in the closing price from the previous day to the current day

feature_cols = ["Return","MA5","MA10","Volatility",
                "RSI","MACD","BB_upper","BB_lower"]

scaler   = MinMaxScaler() # converting the data to a range between 0 and 1 this is important for neural networks to work properly
features = scaler.fit_transform(df[feature_cols])

def make_sequences(data, labels, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(labels[i+lookback])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = make_sequences(features, df["Target"].values, LOOKBACK)

#  5. TRAIN / TEST SPLIT 
split = int((1-TEST_SPLIT) * len(X))
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

# ✅ NEW: Calculate class weights for balanced training
pos_count = np.sum(y_tr)
neg_count = len(y_tr) - pos_count
pos_weight = torch.tensor([neg_count / pos_count]) if pos_count > 0 else torch.tensor([1.0])
print(f"Class weight (for UP days): {pos_weight.item():.3f}")

train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
test_ds  = TensorDataset(torch.tensor(X_te), torch.tensor(y_te))
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

#model
class LSTMClassifier(nn.Module):
    def __init__(self, n_feats, hidden=64):
        super().__init__()
        self.lstm  = nn.LSTM(n_feats, hidden, batch_first=True)
        self.drop  = nn.Dropout(0.3)
        self.fc    = nn.Linear(hidden, 1)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.drop(h[-1])
        return self.fc(out)  # removed sigmoid (will use BCEWithLogitsLoss)

model = LSTMClassifier(n_feats=X.shape[2]).to(DEVICE)

# ✅ NEW: Check if saved model exists and load it
MODEL_PATH = f'stock_model_{TICKER}.pth'
if os.path.exists(MODEL_PATH):
    print(f"🔄 Loading existing model from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("✅ Model loaded! Continuing from previous training.")
else:
    print("🆕 No existing model found. Starting fresh training.")

# ✅ CHANGED: Use BCEWithLogitsLoss with class weights
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ------------------ 7. TRAIN ------------------
print("Starting training...")
best_loss = float('inf')  # Track best loss for saving best model

for epoch in range(1, EPOCHS+1):
    model.train()
    epoch_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb).squeeze()  # squeeze for proper shape
        loss  = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_dl)
    
    # save model if it's the best so far
    if avg_loss < best_loss:
        best_loss = avg_loss
        print(f"New best loss: {avg_loss:.4f} - Saving model...")
        torch.save(model.state_dict(), MODEL_PATH)
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d}/{EPOCHS} - avg loss: {avg_loss:.4f}")

print(f" completed most advanced training {MODEL_PATH}")
print(f" best_loss: {best_loss:.4f}")

# ------------------ 8. EVALUATE ------------------
model.eval()
with torch.no_grad():
    all_preds = []
    all_logits = []  # Let's also track raw logits for debugging
    
    for xb, _ in test_dl:
        xb = xb.to(DEVICE)
        logits = model(xb)  # Don't squeeze here yet
        
        # Handle different output shapes
        if len(logits.shape) > 1:
            logits = logits.squeeze(-1)  # Remove last dimension if needed
        
        probs = torch.sigmoid(logits)  # Apply sigmoid to get probabilities
        
        all_logits.extend(logits.cpu().numpy())
        all_preds.extend(probs.cpu().numpy())

# Debug prints to see what's happening
print(f"Sample logits: {np.array(all_logits)[:10]}")
print(f"Sample probabilities: {np.array(all_preds)[:10]}")
print(f"Min/Max probabilities: {np.min(all_preds):.3f} / {np.max(all_preds):.3f}")

y_pred = (np.array(all_preds) > 0.5).astype(int)

print("\n" + "=" * 50)
print("FINAL RESULTS:")
print("Accuracy:", accuracy_score(y_te, y_pred))
print("\nDetailed Classification Report:")
print(classification_report(y_te, y_pred, digits=4))

# prediction distribution
print("\nPrediction Distribution:")
print(f"Predicted UP: {np.sum(y_pred)} ({np.mean(y_pred)*100:.1f}%)")
print(f"Predicted DOWN: {len(y_pred) - np.sum(y_pred)} ({(1-np.mean(y_pred))*100:.1f}%)")
print("=" * 50)

# save training summary to a log file
with open(f'training_log_{TICKER}.txt', 'a') as f:
    f.write(f"\n{'='*50}\n")
    f.write(f"Training Session: {pd.Timestamp.now()}\n")
    f.write(f"Epochs: {EPOCHS}, LR: {LR}, Best Loss: {best_loss:.4f}\n")
    f.write(f"Accuracy: {accuracy_score(y_te, y_pred):.4f}\n")
    f.write(f"UP predictions: {np.mean(y_pred)*100:.1f}%\n")
    f.write(f"Prob range: {np.min(all_preds):.3f} - {np.max(all_preds):.3f}\n")

print(f" Training session logged to training_log_{TICKER}.txt")

# visualise 
dates = df.index[-len(y_te):]
plt.figure(figsize=(14,6))
plt.plot(dates, y_te,  label="Actual",    marker='o', markersize=3, alpha=0.7)
plt.plot(dates, y_pred, label="Predicted", marker='x', markersize=4, alpha=0.8)
plt.title(f"PyTorch LSTM Trend Prediction for {TICKER} (1=Up,0=Down)")
plt.xlabel("Date"); plt.ylabel("Direction")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()

#  probability plot
plt.figure(figsize=(14,6))
plt.plot(dates, all_preds, label="Prediction Probability", color='red', alpha=0.7)
plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Decision Threshold')
plt.fill_between(dates, 0, all_preds, alpha=0.2, color='red')
plt.title(f"Prediction Probabilities for {TICKER}")
plt.xlabel("Date"); plt.ylabel("Probability of UP")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()

plt.show()