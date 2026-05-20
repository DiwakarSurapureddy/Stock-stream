import importlib
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def load_keras_module():
    for module_name in ("keras", "tensorflow.keras"):
        try:
            return importlib.import_module(module_name)
        except Exception:
            continue
    raise ImportError(
        "Keras is not available. Install with `pip install keras` "
        "or `pip install tensorflow`."
    )


def download_data(ticker: str, years: int) -> pd.DataFrame:
    end = datetime.now()
    start = datetime(end.year - years, end.month, end.day)
    data = yf.download(ticker, start, end, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. Check the symbol or your network."
        )
    return data


def make_sequences(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    x_data, y_data = [], []
    for i in range(window, len(values)):
        x_data.append(values[i - window : i])
        y_data.append(values[i])
    return np.array(x_data), np.array(y_data)


def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, "orange")
    plt.plot(full_data["Close"], "b")
    if extra_data:
        plt.plot(extra_dataset)
    return fig


st.set_page_config(page_title="Stock  Market Data Analyzing", layout="wide")
st.title("Stock Market Data  Analyzing App")
stock = st.text_input("Enter the Stock ID", "GOOG")
years = st.number_input("Years of history", min_value=5, max_value=50, value=25, step=1)
window = st.number_input("LSTM window size", min_value=30, max_value=200, value=100, step=10)
epochs = st.number_input("Training epochs", min_value=1, max_value=50, value=10, step=1)

try:
    data = download_data(stock, years)
except Exception as exc:
    st.error(str(exc))
    st.stop()
    raise SystemExit

st.subheader("Stock Data")
st.dataframe(data)

data["MA_for_100_days"] = data["Close"].rolling(100).mean()
data["MA_for_200_days"] = data["Close"].rolling(200).mean()
data["MA_for_250_days"] = data["Close"].rolling(250).mean()

st.subheader("Original Close Price")
fig = plt.figure(figsize=(15, 6))
plt.plot(data["Close"], color="blue")
st.pyplot(fig)

st.subheader("Original Close Price and MA for 250 days")
st.pyplot(plot_graph((15, 6), data["MA_for_250_days"], data, 0))

st.subheader("Original Close Price and MA for 200 days")
st.pyplot(plot_graph((15, 6), data["MA_for_200_days"], data, 0))

st.subheader("Original Close Price and MA for 100 days")
st.pyplot(plot_graph((15, 6), data["MA_for_100_days"], data, 0))

st.subheader("Original Close Price and MA for 100 days and MA for 250 days")
st.pyplot(
    plot_graph((15, 6), data["MA_for_100_days"], data, 1, data["MA_for_250_days"])
)

splitting_len = int(len(data) * 0.7)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(data[["Close"]])

x_all, y_all = make_sequences(scaled_close, window)
if len(x_all) == 0:
    st.error("Not enough data to build sequences. Reduce the window size.")
    st.stop()

x_all = x_all.reshape((x_all.shape[0], x_all.shape[1], 1))
y_all = y_all.reshape((y_all.shape[0], 1))

train_cut = splitting_len - window
if train_cut <= 0:
    st.error("Not enough data to build sequences. Reduce the window size.")
    st.stop()

x_train, y_train = x_all[:train_cut], y_all[:train_cut]
x_test, y_test = x_all[train_cut:], y_all[train_cut:]

try:
    keras = load_keras_module()
except ImportError as exc:
    st.error(str(exc))
    st.stop()

Sequential = keras.Sequential
LSTM = keras.layers.LSTM
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Input = keras.layers.Input

with st.spinner("Training LSTM model..."):
    model = Sequential(
        [
            Input(shape=(x_train.shape[1], 1)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, epochs=int(epochs), batch_size=32, verbose=0)

predictions = model.predict(x_test, verbose=0)
inv_pred = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_test)

plotting_data = pd.DataFrame(
    {
        "original_test_data": inv_y_test.reshape(-1),
        "predictions": inv_pred.reshape(-1),
    },
    index=data.index[splitting_len:],
)

st.subheader("Original values vs Predicted values")
st.dataframe(plotting_data)

st.subheader("Original Close Price vs Predicted Close price")
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([data["Close"][:splitting_len], plotting_data], axis=0))
plt.legend(["data- not used", "Original Test data", "Predicted data"])
st.pyplot(fig)



