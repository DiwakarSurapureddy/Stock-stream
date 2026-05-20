# Stock Stream

Stock Stream is a Streamlit-based stock analysis application that combines historical market visualization with a simple LSTM forecasting workflow. It downloads historical price data from Yahoo Finance, presents key moving-average views, and trains a neural network on closing prices to compare predicted values against held-out test data.

## Overview

This project is designed as an interactive learning and experimentation tool for market data exploration. Users can enter a stock ticker, choose how many years of history to load, configure the LSTM window size, and control the number of training epochs directly from the app interface.

The application then:

- Downloads historical OHLCV data using `yfinance`
- Displays the raw dataset in a table
- Calculates 100-day, 200-day, and 250-day moving averages
- Visualizes price and moving-average trends with Matplotlib
- Normalizes closing prices and prepares time-series sequences
- Trains an LSTM model using TensorFlow/Keras
- Compares original test values with model predictions

## Features

- Interactive Streamlit dashboard
- Historical stock data retrieval by ticker symbol
- Moving-average trend analysis
- Configurable LSTM sequence window
- Adjustable training epochs
- Prediction-vs-actual comparison table and chart

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- yfinance
- scikit-learn
- TensorFlow / Keras

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## Installation

1. Clone the repository:

```bash
git clone <your-repository-url>
cd "Stock stream"
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

Start the Streamlit development server with:

```bash
streamlit run app.py
```

After launch, Streamlit will provide a local URL in the terminal, typically:

```text
http://localhost:8501
```

## How It Works

1. The app downloads historical stock data for the selected ticker.
2. It computes 100-day, 200-day, and 250-day moving averages from the closing price.
3. Closing prices are scaled to the range `[0, 1]` using `MinMaxScaler`.
4. Sliding windows are created to build supervised training sequences.
5. The dataset is split into training and test segments.
6. A stacked LSTM model is trained on the training data.
7. Predictions are inverse-transformed back to price scale and plotted against actual values.

## Model Notes

- The model is trained every time the app runs with the selected inputs.
- Predictions are based only on historical closing prices.
- This implementation is useful for experimentation and learning, not for production trading systems.
- Results may vary depending on ticker, history length, window size, and training epochs.

## Input Parameters

- `Stock ID`: Ticker symbol such as `GOOG`, `AAPL`, or `MSFT`
- `Years of history`: Number of years of data to download
- `LSTM window size`: Number of previous time steps used for each training sample
- `Training epochs`: Number of passes through the training dataset

## Requirements

The current project dependencies are listed in `requirements.txt`:

- `streamlit==1.57.0`
- `yfinance==0.2.40`
- `matplotlib==3.10.8`
- `numpy==1.26.4`
- `pandas==2.2.2`
- `scikit-learn==1.5.0`
- `tensorflow==2.16.1`

## Limitations

- The model does not save trained weights between sessions.
- Only one input feature, `Close`, is used for forecasting.
- No hyperparameter tuning, validation strategy, or backtesting workflow is included.
- Data availability depends on Yahoo Finance responses and network connectivity.

## Future Improvements

- Add model persistence for faster repeated runs
- Support multiple technical indicators as input features
- Introduce validation metrics such as MAE and RMSE
- Add ticker comparison and sector-level analysis
- Export prediction results and plots

## Disclaimer

This project is for educational and demonstration purposes only. It should not be used as financial advice or as the sole basis for investment decisions.
