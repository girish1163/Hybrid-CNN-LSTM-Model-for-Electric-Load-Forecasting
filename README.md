# Load Forecasting using CNN-LSTM

This project implements a CNN-LSTM neural network for electrical load forecasting using smart grid data.

## 🚀 Features

- **CNN-LSTM Hybrid Model**: Combines convolutional neural networks with LSTM for time series forecasting
- **Data Preprocessing**: Comprehensive data cleaning and normalization
- **Performance Metrics**: MAPE, RMSE, and R² score evaluation
- **Visualization**: Multiple charts for model analysis and comparison
- **Smart Grid Dataset**: 50,000 records of electrical consumption data

## 📊 Dataset

The model uses `smart_grid_dataset.csv` containing:
- Timestamp
- Voltage (V)
- Current (A)
- Power Consumption (kW)
- Reactive Power (kVAR)
- Power Factor
- Solar Power (kW)
- Wind Power (kW)
- Grid Supply (kW)
- Temperature (°C)
- Humidity (%)
- Electricity Price (USD/kWh)
- Predicted Load (kW)

## 🛠️ Requirements

- Python 3.8+
- TensorFlow 2.20.0
- Pandas 2.3.3
- NumPy 2.4.1
- Matplotlib 3.10.8
- Scikit-learn 1.8.0

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/load-forecasting-cnn-lstm.git
cd load-forecasting-cnn-lstm
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
```

## 🏃‍♂️ Usage

1. Ensure the dataset `smart_grid_dataset.csv` is in the project directory
2. Run the model:
```bash
python load_forecasting_cnn_lstm.py
```

## 📈 Results

Current model performance:
- **MAPE**: 67.20%
- **RMSE**: 0.00 MW
- **R² Score**: -0.0002

## 📁 Project Structure

```
load-forecasting-cnn-lstm/
├── load_forecasting_cnn_lstm.py    # Main model script
├── smart_grid_dataset.csv           # Dataset
├── cnn_lstm_load_forecast.h5        # Trained model
├── load_forecast_results.csv        # Prediction results
├── actual_vs_predicted.png          # Visualization
├── training_history.png             # Training history
├── .gitignore                       # Git ignore file
└── README.md                        # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔍 Model Architecture

The CNN-LSTM model consists of:
- **Conv1D layers**: For feature extraction
- **MaxPooling1D**: For dimensionality reduction
- **LSTM layers**: For temporal pattern learning
- **Dropout**: For regularization
- **Dense layers**: For final prediction

## 📊 Visualizations

The model generates several visualizations:
- Actual vs Predicted Load
- Training History
- Error Distribution
- Model Comparison Charts

## 🚧 Future Improvements

- [ ] Hyperparameter tuning
- [ ] Feature engineering
- [ ] Ensemble methods
- [ ] Real-time prediction
- [ ] Web interface

## 📧 Contact

Your Name - [@your_twitter](https://twitter.com/your_twitter)

Project Link: [https://github.com/YOUR_USERNAME/load-forecasting-cnn-lstm](https://github.com/YOUR_USERNAME/load-forecasting-cnn-lstm)
