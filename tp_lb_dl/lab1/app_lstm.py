"""
Streamlit LSTM Stock Price Predictor
-----------------------------------
- Downloads stock data from yfinance
- Handles 'Adj Close' or 'Close' columns automatically
- Prepares data for LSTM model
- Predicts and forecasts stock prices

Run:
    streamlit run app_lstm.py

Requirements:
    pip install streamlit yfinance pandas numpy scikit-learn tensorflow plotly matplotlib
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go

# ---------------------- Streamlit Page Config ----------------------
st.set_page_config(page_title="LSTM Stock Predictor", layout="wide")

# ---------------------- Helper Functions ----------------------
@st.cache_data(show_spinner=False)
def download_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download stock data and handle missing Adj Close column."""
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError(f"No data found for {ticker} between {start_date} and {end_date}")

    # Handle both Adj Close and Close
    if 'Adj Close' in df.columns:
        df = df[['Adj Close']].rename(columns={'Adj Close': 'adj_close'})
    elif 'Close' in df.columns:
        df = df[['Close']].rename(columns={'Close': 'adj_close'})
    else:
        raise ValueError(f"No 'Adj Close' or 'Close' column found for {ticker}")

    df.index = pd.to_datetime(df.index)
    return df


def create_sequences(values: np.ndarray, look_back: int):
    """Create X, y sequences from time series."""
    X, y = [], []
    for i in range(len(values) - look_back):
        X.append(values[i:i+look_back])
        y.append(values[i+look_back])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape, units=50, dropout=0.2):
    """Build an LSTM model."""
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units // 2, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def plot_actual_vs_pred(dates, actual, predicted, title="Actual vs Predicted"):
    """Plot actual vs predicted values."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual.flatten(), name='Actual'))
    fig.add_trace(go.Scatter(x=dates, y=predicted.flatten(), name='Predicted'))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)


# ---------------------- Streamlit UI ----------------------
st.title("📈 LSTM Stock Price Predictor (yfinance + Streamlit)")

with st.sidebar:
    st.header("⚙️ Configuration")
    ticker = st.text_input("Ticker", value='AAPL').upper()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=pd.to_datetime('2018-01-01'))
    with col2:
        end_date = st.date_input("End date", value=pd.to_datetime('2024-12-31'))

    look_back = st.slider('Look-back window (days)', 10, 200, 60)
    test_size = st.slider('Test set proportion', 0.05, 0.5, 0.2, step=0.05)
    epochs = st.number_input('Training epochs', 1, 500, 30)
    batch_size = st.number_input('Batch size', 8, 512, 32)
    patience = st.number_input('Early stopping patience', 0, 50, 5)
    forecast_days = st.number_input('Forecast next N days', 1, 60, 7)
    st.markdown("---")
    show_raw = st.checkbox('Show raw data')

# ---------------------- Main Execution ----------------------
if st.button('🚀 Run Prediction'):
    try:
        df = download_data(ticker, str(start_date), str(end_date))
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        st.stop()

    if show_raw:
        st.write(df.tail(10))

    # Prepare data
    data = df[['adj_close']].values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    test_len = int(len(data_scaled) * test_size)
    train_len = len(data_scaled) - test_len

    train_data = data_scaled[:train_len]
    test_data = data_scaled[train_len - look_back:]

    X_train, y_train = create_sequences(train_data, look_back)
    X_test, y_test = create_sequences(test_data, look_back)

    # reshape for LSTM [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    st.write(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")

    # Build and train model
    model = build_lstm_model((look_back, 1), units=64, dropout=0.2)
    early = EarlyStopping(monitor='val_loss', patience=int(patience), restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=int(epochs),
        batch_size=int(batch_size),
        callbacks=[early],
        verbose=0
    )

    # Predictions
    preds = model.predict(X_test)
    preds_rescaled = scaler.inverse_transform(preds)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = sqrt(mean_squared_error(y_test_rescaled, preds_rescaled))
    st.success(f"✅ Test RMSE: {rmse:.4f}")

    # Plot test segment
    test_dates = df.index[train_len:]
    plot_dates = test_dates[look_back:look_back+len(preds_rescaled)]
    plot_actual_vs_pred(plot_dates, y_test_rescaled, preds_rescaled, title=f"{ticker} - Actual vs Predicted")

    # Forecast next N days
    last_window = data_scaled[-look_back:]
    forecast_scaled = []
    current_window = last_window.copy()

    for _ in range(int(forecast_days)):
        x_input = current_window.reshape((1, look_back, 1))
        yhat = model.predict(x_input)
        forecast_scaled.append(yhat[0, 0])
        current_window = np.append(current_window[1:], yhat, axis=0)

    forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
    forecast_rescaled = scaler.inverse_transform(forecast_scaled).flatten()
    future_dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=int(forecast_days))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-200:], y=df['adj_close'].values[-200:], name='Recent Actual'))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast_rescaled, name=f'Forecast next {forecast_days} days'))
    fig.update_layout(title=f"{ticker} - {forecast_days}-Day Forecast", xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)

    # Training loss
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=history.history['loss'], name='train_loss'))
    if 'val_loss' in history.history:
        fig2.add_trace(go.Scatter(y=history.history['val_loss'], name='val_loss'))
    fig2.update_layout(title='Training Loss', xaxis_title='Epoch', yaxis_title='Loss')
    st.plotly_chart(fig2, use_container_width=True)

    # Show forecast
    st.subheader('🔮 Forecasted Values')
    forecast_df = pd.DataFrame({'date': future_dates, 'forecast': forecast_rescaled})
    forecast_df = forecast_df.set_index('date')
    st.dataframe(forecast_df)

    st.info("Note: This model is for educational purposes only. Do not use for trading decisions.")

else:
    st.write("👈 Set parameters on the left and click **Run Prediction**.")
