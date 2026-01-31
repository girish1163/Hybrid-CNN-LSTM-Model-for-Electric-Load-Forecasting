import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import math

# Load the dataset
data = pd.read_csv('smart_grid_dataset.csv')

# Convert Timestamp column to datetime format and set as index
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data = data.set_index('Timestamp')

# Use 'Power Consumption (kW)' as our target variable (load)
# Convert from kW to MW by dividing by 1000
load_data = (data['Power Consumption (kW)'] / 1000).values.reshape(-1, 1)

# Print basic information about the data
print("\nFirst few rows of the dataset:")
print(data.head())
print("\nDataset shape:", data.shape)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(load_data)

# Function to create sequences for time series
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

# Define sequence length (number of time steps to look back)
seq_length = 24  # Using 24 hours as one sequence
X, y = create_sequences(scaled_data, seq_length)

# Reshape input to be [samples, time steps, features] for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build CNN-LSTM model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, 1)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions back to original scale
train_predict = scaler.inverse_transform(train_predict)
y_train_actual = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test_actual = scaler.inverse_transform([y_test])

# Calculate performance metrics
def calculate_metrics(actual, predicted):
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    return mape, rmse, r2

# Calculate metrics for test set
mape, rmse, r2 = calculate_metrics(y_test_actual[0], test_predict[:, 0])

# Print performance metrics
print("\nPerformance Metrics:")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²) Score: {r2:.4f}")

# Plot actual vs predicted values for test set
plt.figure(figsize=(15, 6))
plt.plot(y_test_actual[0], label='Actual Load')
plt.plot(test_predict[:, 0], label='Predicted Load')
plt.title('Actual vs Predicted Electrical Load')
plt.xlabel('Time (hours)')
plt.ylabel('Load (MW)')
plt.legend()
plt.savefig('actual_vs_predicted.png')
plt.show()

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.savefig('training_history.png')
plt.show()

# Save the model
model.save('cnn_lstm_load_forecast.h5')

# Create a DataFrame with actual and predicted values for test set
results = pd.DataFrame({
    'Actual_Load': y_test_actual[0],
    'Predicted_Load': test_predict[:, 0],
    'Error': y_test_actual[0] - test_predict[:, 0]
})

# Save results to CSV
results.to_csv('load_forecast_results.csv', index=False)
print("\nResults saved to 'load_forecast_results.csv'")
print("Model saved as 'cnn_lstm_load_forecast.h5'")
