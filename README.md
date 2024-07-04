## Stock Prediction Project using LSTM

This project demonstrates a stock prediction model built using a Long Short-Term Memory (LSTM) neural network. The project is designed to analyze historical stock data and generate predictions for future prices.

### Project Structure

The project is organized as follows:

```
└── src
    └── LongShortTermMemory.py
    └── StockDataProcessor.py
    └── StockDataVisualizer.py
    └── StockPredictionConfig.py (Deprecated)
    └── __init__.py
```

### Dependencies

The following Python libraries are mainly required for this project:

- **tensorflow**
- **pandas**
- **numpy**
- **matplotlib**
- **yfinance**

You can install these dependencies using `pip`:

```bash
pip install tensorflow pandas numpy matplotlib yfinance
```

Prefer to use the `environment.yaml` file and setting up python conda GPU container.

### Configuration

**Class been deprecated, but parameters are still used outside a class context.**
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

The `train` function in `LongShortTermMemory.py` trains the LSTM model:

- **Loads data:** Reads historical stock data.
- **Creates the model:** Builds the LSTM model.
- **Compiles the model:** Sets up the model with an optimizer, loss function, and metrics.
- **Trains the model:** Fits the model to the training data.
- **Saves the model weights:** Stores the trained weights for later use.
- **Evaluates the model:** Tests the model's performance on the test data.

### Prediction

The `infer` function in `LongShortTermMemory.py` loads the trained model and predicts future stock prices:

- **Loads model weights:** Loads the saved model weights.
- **Generates future data:** Simulates future stock data.
- **Makes predictions:** Uses the loaded model to predict future stock prices.
- **Plots predictions:** Visualizes the predicted prices against the actual or simulated prices.

### Usage

You can find the ```stock_prediction_lstm.ipynb``` notebook with useful examples for model creation, traning and prediction.

### Visualizations

The project includes various visualizations:

- **Data split:** Plots the initial data split into training and validation sets.
- **Loss:** Plots the training loss and validation loss over epochs.
- **MSE:** Plots the training MSE and validation MSE over epochs.
- **Predictions:** Plots the predicted prices against the actual or simulated prices.
- **Future predictions:** Plots predicted future prices against simulated prices.

### Hyperparameter Tuning

The project includes a separate notebook (`hyper_tuning.ipynb`) dedicated to hyperparameter tuning for the LSTM model. This notebook uses scikit-learn's `GridSearchCV` or `RandomizedSearchCV` to find the best combination of hyperparameters for improved model performance.

### Cross Validation

The project includes a separate notebook (`cross_validation.ipynb`) dedicated to cross validation operation for the LSTM model. This notebook uses scikit-learn and keras for testong the model performance.

