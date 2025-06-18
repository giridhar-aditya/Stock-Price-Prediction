# 🚀 AAPL Stock Price Prediction using Deep Learning

## 📊 Project Overview

This project leverages cutting-edge deep learning techniques to forecast Apple's stock prices with high accuracy. Utilizing a hybrid architecture combining CNN, BiLSTM, and Transformer models, we capture both short-term patterns and long-term dependencies in financial time series data.

---

## 🗂 Dataset Details

* **Data Source:** Yahoo Finance (AAPL)
* **Time Period:** 2010-02-22 to Present
* **Total Records:** 3405 data points

### 🏷 Features Used:

* 📈 Price Data: Open, High, Low, Close, Adj Close
* 📊 Volume Data: Volume
* 📉 Technical Indicators:

  * SMA\_20 (Simple Moving Average)
  * EMA\_20 (Exponential Moving Average)
  * RSI (Relative Strength Index)
  * MACD (Moving Average Convergence Divergence)
  * Bollinger Bands (High, Low, Width)
  * Stochastic Oscillator (K, D)
  * ADX (Average Directional Index)
  * CCI (Commodity Channel Index)
  * ATR (Average True Range)

---

## 🧠 Model Architecture

### 🔬 Hybrid Deep Learning Model:

* 🧩 **CNN Layer:** Extracts local temporal features from sequences.
* 🔄 **BiLSTM Layer:** Captures bidirectional long-term dependencies.
* ✨ **Transformer Encoder:** Models complex relationships via attention mechanisms.
* 🧮 **Fully Connected Layers:** Outputs final stock price predictions.

---

## ⚙️ Model Training Configuration

* **Input Sequence Length:** 60 days
* **Train/Test Split:** 80% Train, 20% Test
* **Loss Function:** Mean Squared Error (MSE)
* **Optimizer:** Adam
* **Batch Size:** 32
* **Epochs:** 50
* **Hardware:** GPU/CPU Compatible

---

## 📈 Evaluation Results

After careful training and optimization, the model achieved the following results on the test dataset:

| 📊 Metric | 🔢 Value |
| --------- | -------- |
| MAE       | 5.9283   |
| RMSE      | 11.0309  |
| R² Score  | 0.9605   |
| Accuracy  | 90.33%   |

✅ **Outstanding predictive power with 90%+ accuracy and excellent fit (R² = 0.96).**

---

## 📉 Visualization

![Prediction Graph](./prediction_plot.png)

The graph illustrates the close alignment between actual and predicted stock prices, demonstrating the model's reliability.

---

## 🚀 How to Use

1. **Clone the repository**
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Train the model:** `python train.py`
4. **Evaluate performance:** `python evaluate.py`

Pre-trained model is saved as `model.pth` for immediate evaluation.

---

## 🔮 Potential Future Enhancements

* 📈 Incorporate macroeconomic indicators.
* 📰 Integrate financial news sentiment analysis.
* 📆 Extend to multi-horizon forecasting.
* 📊 Add explainable AI (XAI) techniques.
