import os
import pandas as pd

from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from src.StockDataProcessor import StockDataProcessor
from src.LongShortTermMemory import LSTMModel

# %%
FOLDER_PREFIX = "data/min/"
STOCK_START_DATE = pd.to_datetime("2017-06-07 15:59:00")
STOCK_VALIDATION_DATE = pd.to_datetime("2024-06-09 09:30:00")
STOCK_END_DATE = pd.to_datetime("2024-06-12 15:59:00")
TIME_STEPS = 60
TOKEN = "GOOG"
RUN_FOLDER = f"{FOLDER_PREFIX}{TOKEN}/"
WORK_DIR = os.path.join(os.getcwd(), RUN_FOLDER)
CSV_FILE = f"{WORK_DIR}data.csv"
PROJECT_FOLDER = os.path.join(os.getcwd(), RUN_FOLDER)
if not os.path.exists(PROJECT_FOLDER):
    os.makedirs(PROJECT_FOLDER)

# %%
(x_train, y_train), (x_test, y_test), (training_data, test_data) = StockDataProcessor.load_csv_transform_to_numpy(TIME_STEPS, CSV_FILE, STOCK_VALIDATION_DATE)

# %%
lstm = LSTMModel(PROJECT_FOLDER)

# Define the parameter grid for the search
param_grid = {
    'model__units': [50, 100, 150, 200],
    'model__dropout': [0.2, 0.3, 0.4],
    'model__activation': ['relu', 'tanh', 'sigmoid'],
    'model__optimizer': ['adam', 'rmsprop', 'sgd'],
    'model__batch_size': [5, 10, 20, 40],
    'model__epochs': [50, 100, 150]
}

# %%
# Define a function to create the LSTM model
def create_model(units=100, dropout=0.2, activation='relu', optimizer='adam'):
    """
    Creates the LSTM model with specified hyperparameters.

    Args:
        units: The number of units in each LSTM layer.
        dropout: The dropout rate.
        activation: The activation function.
        optimizer: The optimizer to use.

    Returns:
        tf.keras.Model: The compiled LSTM model.
    """

    model = Sequential()
    # 1st layer with Dropout regularisation
    # * units = add 100 neurons is the dimensionality of the output space
    # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
    # * input_shape => Shape of the training dataset
    model.add(LSTM(units=units, return_sequences=True)
    # 20% of the layers will be dropped
    model.add(Dropout(dropout))
    # 2nd LSTM layer
    # * units = add 50 neurons is the dimensionality of the output space
    # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
    model.add(LSTM(units=units // 2, return_sequences=True))
    # 20% of the layers will be dropped
    model.add(Dropout(dropout))
    # 3rd LSTM layer
    # * units = add 50 neurons is the dimensionality of the output space
    # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
    model.add(LSTM(units=units // 2, return_sequences=True))
    # 50% of the layers will be dropped
    model.add(Dropout(dropout * 2.5))
    # 4th LSTM layer
    # * units = add 50 neurons is the dimensionality of the output space
    model.add(LSTM(units=units // 2))
    # 50% of the layers will be dropped
    model.add(Dropout(dropout * 2.5))
    # Dense layer that specifies an output of one unit
    model.add(Dense(units=1, activation=activation))  # Apply activation to the output layer

    if optimizer == 'adam':
        optimizer = Adam()
    elif optimizer == 'rmsprop':
        optimizer = RMSprop()
    elif optimizer == 'sgd':
        optimizer = SGD()
    else:
        raise ValueError(f"Invalid optimizer: {optimizer}. Choose from 'adam', 'rmsprop', or 'sgd'.")

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[MeanSquaredError(name='MSE')])
    return model

# %%
# Create the KerasRegressor
model = KerasRegressor(model=create_model, verbose=0)

# Create the GridSearchCV object
# grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=1)  # Use GridSearchCV
# OR
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=3, n_jobs=-1)  # Use RandomizedSearchCV

# Fit the GridSearchCV to the data
grid_result = grid.fit(x_train, y_train)

# %%
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Use the best parameters to create a model for final evaluation
# best_params = grid_result.best_params_
# best_model = create_model(**best_params)  # Use the best parameters to create the model
# best_model.fit(x_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'])