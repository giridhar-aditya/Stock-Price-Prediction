# Stock Price Prediction with LSTM and Linear Regression

## 📌 Overview
This project predicts stock closing prices using **deep learning (LSTM)** and **Linear Regression** models. It automatically downloads historical stock data using the `yfinance` library and evaluates model performance through key metrics and visualizations.

## 🧾 Data Pipeline

### `make_data.py`
- Downloads historical stock data using `yfinance`
- Symbol: `AAPL` (configurable)
- Date range: `2019-06-01` to `2024-06-01`
- Saves data to `stock_data.csv`

```bash
python "make data.py"
