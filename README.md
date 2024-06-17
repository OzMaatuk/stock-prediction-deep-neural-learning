## Stock Prediction Project using LSTM

This project demonstrates a stock prediction model built using a Long Short-Term Memory (LSTM) neural network. The project is designed to analyze historical stock data and generate predictions for future prices.

### Project Structure

The project is organized as follows:

```
└── src
    └── InferLstmModel.py
    └── LongShortTermMemory.py
    └── StockDataProcessor.py
    └── StockDataVisualizer.py
    └── StockPredictionConfig.py
    └── TrainLstmNetwork.py
    └── __init__.py
    └── __pycache__
        └── Infer.cpython-311.pyc
        └── InferLstmModel.cpython-311.pyc
        └── LongShortTermMemory.cpython-311.pyc
        └── StockDataProcessor.cpython-311.pyc
        └── StockDataVisualizer.cpython-311.pyc
        └── StockPredictionConfig.cpython-311.pyc
        └── TrainLstmNetwork.cpython-311.pyc
        └── __init__.cpython-311.pyc
```

### Dependencies

The following Python libraries are required for this project:

- **tensorflow**
- **pandas**
- **numpy**
- **matplotlib**
- **yfinance**

You can install these dependencies using `pip`:

```bash
pip install tensorflow pandas numpy matplotlib yfinance
```

### Configuration

The configuration parameters for the project are stored in the `StockPredictionConfig` class. These parameters include:

- **`ticker`:** The stock ticker symbol (e.g., "GOOG").
- **`start_date`:** The start date for data retrieval.
- **`end_date`:** The end date for data retrieval.
- **`validation_date`:** The date to split the data into training and validation sets.
- **`project_folder`:** The path to the project folder where results will be saved.
- **`epochs`:** The number of epochs for training the LSTM model.
- **`time_steps`:** The number of time steps to use for the LSTM model.
- **`token`:** A unique identifier for the project run.
- **`batch_size`:** The batch size for training the LSTM model.

### Data Processing

The `StockDataProcessor` class handles data loading, preprocessing, and transformation. It includes methods for:

- **Downloading data:** Downloads historical stock data from Yahoo Finance.
- **Loading CSV data:** Loads data from a CSV file.
- **Transforming data:** Converts data into NumPy arrays for training and testing.
- **Generating future data:** Simulates future stock data based on the latest close price.

### Model Training

The `TrainLstmNetwork` function in `TrainLstmNetwork.py` trains the LSTM model:

- **Loads data:** Reads historical stock data.
- **Creates the model:** Builds the LSTM model.
- **Compiles the model:** Sets up the model with an optimizer, loss function, and metrics.
- **Trains the model:** Fits the model to the training data.
- **Saves the model weights:** Stores the trained weights for later use.
- **Evaluates the model:** Tests the model's performance on the test data.

### Prediction

The `InferLstmModel` function in `InferLstmModel.py` loads the trained model and predicts future stock prices:

- **Loads model weights:** Loads the saved model weights.
- **Generates future data:** Simulates future stock data.
- **Makes predictions:** Uses the loaded model to predict future stock prices.
- **Plots predictions:** Visualizes the predicted prices against the actual or simulated prices.

### Usage

To use the project:

1. **Install dependencies:**
   ```bash
   pip install tensorflow pandas numpy matplotlib yfinance
   ```

2. **Run `TrainLstmNetwork` to train the model:**
   ```python
   from src import StockPredictionConfig
   from src import StockDataProcessor
   from src import TrainLstmNetwork

   stock_config = StockPredictionConfig(
       ticker="GOOG",
       start_date=pd.to_datetime("2017-06-07 15:59:00"),
       end_date=pd.to_datetime("2024-06-12 15:59:00"),
       validation_date=pd.to_datetime("2024-06-09 09:30:00"),
       epochs=100,
       time_steps=60,
       token="GOOG"
   )

   data_processor = StockDataProcessor()
   TrainLstmNetwork.TrainLstmNetwork(stock_config, x_train, y_train, x_test, y_test, training_data, test_data)
   ```

3. **Run `InferLstmModel` to make predictions:**
   ```python
   from src import StockDataProcessor
   from src import InferLstmModel

   start_date = pd.to_datetime("2024-06-13 09:30:00")
   end_date = pd.to_datetime("2024-06-14 15:59:00")
   latest_close_price = 120.00  # Replace with the actual latest close price

   InferLstmModel.InferLstmModel(start_date, end_date, latest_close_price, WORK_DIR, TIME_STEPS, STOCK_TICKER, "USD") 
   ```

### Visualizations

The project includes various visualizations:

- **Data split:** Plots the initial data split into training and validation sets.
- **Loss:** Plots the training loss and validation loss over epochs.
- **MSE:** Plots the training MSE and validation MSE over epochs.
- **Predictions:** Plots the predicted prices against the actual or simulated prices.
- **Future predictions:** Plots predicted future prices against simulated prices.

### Notes

- The model's performance and accuracy depend on various factors, including the quality and quantity of data, the model architecture, and the hyperparameters used. 
- This project provides a basic framework for stock prediction. You can enhance it by:
    - Adding more features (e.g., technical indicators, news sentiment).
    - Experimenting with different model architectures.
    - Tuning hyperparameters using techniques like grid search or Bayesian optimization.
    - Implementing more advanced data processing techniques.

Let me know if you have any questions or need further assistance with this project. 


