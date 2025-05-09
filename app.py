import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Streamlit title and model selector
st.title("Time Series Forecast Visualization")

model_choice = st.selectbox(
    "Select Model for Prediction",
    options=["ARIMA", "SARIMA", "LSTM"]
)

# Common data
data = pd.read_csv('all_data.csv', parse_dates=['timestamp'], index_col='timestamp')

# ------------------ ARIMA/SARIMA Block ------------------
if model_choice in ["ARIMA", "SARIMA"]:
    # Load models
    with open('models/arima_model.pkl', 'rb') as f:
        arima_model_fit = pickle.load(f)

    with open('models/sarima_model_7.pkl', 'rb') as f:
        sarima_model_fit = pickle.load(f)

    # Prepare train/test
    train = data[:round(len(data)*80/100)]
    test = data[round(len(data)*80/100):]

    # Add February 2015 to test
    feb_dates = pd.date_range('2015-02-01', periods=28)
    new_data = pd.DataFrame({'value': 0}, index=feb_dates)
    test = pd.concat([test, new_data])

    # Add null values to original data for Feb
    new_data_null = pd.DataFrame({'value': np.nan}, index=feb_dates)
    data = pd.concat([data, new_data_null])

    # Set date range
    min_date = test.index[0].to_pydatetime()
    max_date = test.index[-1].to_pydatetime()

    # Slider for prediction end date
    prediction_end = st.slider(
        "Select End Date for Prediction",
        min_value=min_date,
        max_value=max_date,
        value=max_date,
        format="YYYY-MM-DD"
    )

    # Prediction
    if model_choice == "ARIMA":
        prediction = arima_model_fit.predict(start=test.index[0], end=prediction_end, typ='levels')
        model_label = "ARIMA Prediction"
        model_color = 'red'
        model_linestyle = '--'
    else:
        prediction = sarima_model_fit.predict(start=test.index[0], end=prediction_end, typ='levels')
        model_label = "SARIMA Prediction"
        model_color = 'green'
        model_linestyle = '-.'

    # Add prediction to data
    data.loc[test.index[:len(prediction)], 'prediction'] = prediction

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(data.index, data['value'], label="Actual", color='blue')
    ax.plot(test.index[:len(prediction)], prediction, label=model_label, color=model_color, linestyle=model_linestyle)
    ax.axvline(x=test.index[0], color='black', linestyle=':', label='Train/Test Split')
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(f"{model_choice} Prediction vs Actual")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    result_df = pd.DataFrame({
        'Time': test.index,
        'Actual': data.loc[test.index, 'value'].values,
        'Prediction': prediction.reindex(test.index).values
    }).dropna(subset=['Prediction'])

    result_df = result_df[result_df['Time'] <= prediction_end]

    st.subheader("Prediction vs Actual Table")
    st.dataframe(result_df.set_index('Time').round(2))

# ------------------ LSTM Block ------------------
else:
    df = data.copy()
    feb_dates = pd.date_range('2015-02-01', '2015-02-28')
    new_data = pd.DataFrame({'value': np.nan}, index=feb_dates)
    df = pd.concat([df, new_data])

    # Scale the data (keep both versions: with and without NaNs)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values_with_nan = scaler.fit_transform(df['value'].values.reshape(-1, 1))
    scaled_values_no_nan = np.nan_to_num(scaled_values_with_nan, nan=0.0)

    train_size = int(len(scaled_values_no_nan) * 0.7)
    train_data = scaled_values_no_nan[:train_size]
    test_data = scaled_values_no_nan[train_size:]

    def create_sequences(data, original_data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            actual = original_data[i + sequence_length][0]
            y.append(actual if not np.isnan(actual) else np.nan)
        return np.array(X), np.array(y)

    sequence_length = 10
    X_train, y_train = create_sequences(train_data, scaled_values_with_nan[:train_size], sequence_length)
    X_test, y_test = create_sequences(test_data, scaled_values_with_nan[train_size:], sequence_length)

    # Load LSTM model
    model = load_model('models/lstm_model.h5')

    # Predict and inverse transform
    predictions = model.predict(X_test)
    predicted_values = scaler.inverse_transform(predictions)

    actual_values = np.where(
        np.isnan(y_test.reshape(-1, 1)),
        np.nan,
        scaler.inverse_transform(y_test.reshape(-1, 1))
    )

    prediction_dates = df.index[train_size + sequence_length:]

    # Slider for selecting prediction range
    min_date = prediction_dates[0].to_pydatetime()
    max_date = prediction_dates[-1].to_pydatetime()

    prediction_end = st.slider(
        "Select End Date for Prediction",
        min_value=min_date,
        max_value=max_date,
        value=max_date,
        format="YYYY-MM-DD"
    )

    # Filter predictions and actuals
    filtered_dates = prediction_dates[prediction_dates <= prediction_end]
    filtered_preds = predicted_values[:len(filtered_dates)]
    filtered_actuals = actual_values[:len(filtered_dates)]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['value'], label='Actual', color='blue')
    ax.plot(filtered_dates, filtered_preds, label='LSTM Prediction', color='red', linestyle='--')
    ax.axvline(x=df.index[train_size], color='black', linestyle=':', label='Train/Test Split')
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title("LSTM Prediction vs Actual")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Display table
    result_df = pd.DataFrame({
        'Date': filtered_dates,
        'Prediction': filtered_preds.flatten(),
        'Actual': filtered_actuals.flatten()
    }).set_index('Date')

    st.subheader("Prediction Table")
    st.dataframe(result_df)
