import os
import tensorflow as tf
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


from src.StockDataVisualizer import StockDataVisualizer
from src.StockPredictionConfig import StockPredictionConfig
from src.StockDataProcessor import StockDataProcessor


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.metrics import MeanSquaredError # RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
# from tensorflow.keras.regularizers import l1, l2
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.callbacks import LearningRateScheduler

class LSTMModel:
    def __init__(self, project_folder):
        self.project_folder = project_folder

    @staticmethod
    def get_metrics():
        return [MeanSquaredError(name='MSE')]
    
        # Can also add more metrics, for example:
        # return [
        #     MeanSquaredError(name='MSE'),
        #     RootMeanSquaredError(),
        #     MeanAbsoluteError() 
        # ]

    @staticmethod
    def get_callbacks():
        callback = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1) # kernel_regularizer, recurrent_regularizer
        return [callback]
    
        # Can also add LearningRateScheduler to callbacks, for example:
        # def scheduler(epoch, lr):
        #     if epoch < 10:  # Initial decay phase
        #         return lr * 0.95 ** epoch # Exponential decay
        #     else:
        #         return lr * 0.99  # Slower linear decay
        # lr_scheduler = LearningRateScheduler(scheduler)

        # return [early_stopping, lr_scheduler]  # Return both callbacks


    @staticmethod
    def create(units=100, dropout=0.2, activation='relu', optimizer='adam'):
        """
        Creates the LSTM model with specified hyperparameters.

        Args:
            x_train: The training data.
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
        model.add(LSTM(units=units, return_sequences=True))
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
        model.add(Dense(units=1, activation=activation))

        # Can also add Regularization to the LSTM layer
        # kernel_regularizer=l2(kernel_regularizer), 
        # OR / AND
        # recurrent_regularizer=l1(recurrent_regularizer),
        # OR / AND add batch Normalization layer after LSTM layer 
        # model.add(BatchNormalization())

        if optimizer == 'adam':
            optimizer = Adam()
        elif optimizer == 'rmsprop':
            optimizer = RMSprop()
        elif optimizer == 'sgd':
            optimizer = SGD()
        else:
            raise ValueError(f"Invalid optimizer: {optimizer}. Choose from 'adam', 'rmsprop', or 'sgd'.")

        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=LSTMModel.get_metrics())
        return model


class LSTMModel:
    def __init__(self, project_folder):
        self.project_folder = project_folder

    @staticmethod
    def get_metrics():
        return [MeanSquaredError(name='MSE')]

    @staticmethod
    def get_callbacks():
        callback = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)
        return [callback]

    @staticmethod
    def create(units=100, dropout=0.2, activation='relu', optimizer='adam'):
        """Creates the LSTM model.

        Args:
            units (int, optional): Number of LSTM units. Defaults to 100.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
            activation (str, optional): Activation function. Defaults to 'relu'.
            optimizer (str, optional): Optimizer. Defaults to 'adam'.

        Returns:
            tf.keras.Model: The compiled LSTM model.

        Raises:
            ValueError: If an invalid optimizer is provided.
        """
        model = Sequential()
        model.add(LSTM(units=units, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(units=units // 2, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(units=units // 2, return_sequences=True))
        model.add(Dropout(dropout * 2.5))
        model.add(LSTM(units=units // 2))
        model.add(Dropout(dropout * 2.5))
        model.add(Dense(units=1, activation=activation))

        optimizers = {
            'adam': Adam(),
            'rmsprop': RMSprop(),
            'sgd': SGD()
        }
        if optimizer.lower() in optimizers:
            optimizer = optimizers[optimizer.lower()]
        else:
            raise ValueError(f"Invalid optimizer: {optimizer}. Choose from 'adam', 'rmsprop', or 'sgd'.")

        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=LSTMModel.get_metrics())
        return model

    @staticmethod
    def train(stock_config: StockPredictionConfig, x_train, y_train, x_test, y_test, training_data, test_data) -> None:
        """Trains the LSTM model, saves weights, and evaluates performance.

        Args:
            stock_config (StockPredictionConfig): Configuration settings.
            x_train (np.array): Training data.
            y_train (np.array): Training labels.
            x_test (np.array): Testing data.
            y_test (np.array): Testing labels.
            training_data (pd.DataFrame):  Training DataFrame (for plotting).
            test_data (pd.DataFrame): Testing DataFrame (for plotting).
        """
        try:
            lstm = LSTMModel(stock_config.project_folder)
            model = lstm.create()
            history = model.fit(
                x_train, y_train,
                epochs=stock_config.epochs,
                batch_size=stock_config.batch_size,
                validation_data=(x_test, y_test),
                callbacks=lstm.get_callbacks()
            )

            model_path = os.path.join(stock_config.project_folder, 'model_weights.keras')
            model.save(model_path)
            logging.info(f"Model weights saved to: {model_path}")

            lstm.evaluate(model, x_test, y_test)
            StockDataVisualizer.plot_results(stock_config, history, training_data, test_data, model, x_test)
            logging.info("Training and plotting complete.")

        except Exception as e:
            logging.error(f"An error occurred during training: {e}")


    @staticmethod
    def evaluate(model, x_test, y_test):
        """Evaluates the model on test data and logs the results.

        Args:
            model: The trained LSTM model.
            x_test (np.array): Testing data.
            y_test (np.array): Testing labels.
        """
        try:
            baseline_results = model.evaluate(x_test, y_test, verbose=2)
            for name, value in zip(model.metrics_names, baseline_results):
                logging.info(f"Evaluation - {name}: {value}")

        except Exception as e:
            logging.error(f"An error occurred during evaluation: {e}")



    @staticmethod
    def infer(start_date, end_date, latest_close_price, work_dir, time_steps):
        """Generates and plots future predictions.

        Args:
            start_date (str): Prediction start date ('YYYY-MM-DD').
            end_date (str): Prediction end date ('YYYY-MM-DD').
            latest_close_price (float): The last known closing price.
            work_dir (str):  Working directory.
            time_steps (int): Number of time steps.
        """
        try:
            x_test, y_test, test_data = StockDataProcessor.generate_future_data(
                time_steps, start_date, end_date, latest_close_price
            )

            test_data = test_data[:-1]

            if x_test.shape[0] > 0:
                model_path = os.path.join(work_dir, 'model_weights.keras')
                model = tf.keras.models.load_model(model_path)
                model.summary()

                predicted_data = model.predict(x_test)
                predicted_data = StockDataProcessor.min_max.inverse_transform(predicted_data)
                predicted_data = pd.DataFrame(predicted_data, columns=['Predicted'])

                if StockDataProcessor.is_day_interval(start_date, end_date):
                    predicted_dates = pd.date_range(start=test_data.index[0], end=test_data.index[-1]) #  periods=len(test_predictions_baseline))
                else:
                    predicted_dates = pd.date_range(start=test_data.index[0], periods=len(predicted_data), freq="1min")

                predicted_data['Datetime'] = predicted_dates
                test_data['Datetime'] = predicted_dates
                predicted_data.reset_index(drop=True, inplace=True)
                test_data.reset_index(drop=True, inplace=True)
                test_data.set_index('Datetime', inplace=True)
                predicted_data.set_index('Datetime', inplace=True)
                # Plotting predictions
                StockDataVisualizer.plot_future(test_data, predicted_data, work_dir)
            else:
                logging.warning("Future data is empty. No predictions generated.")

        except Exception as e:
            logging.error(f"An error occurred during inference: {e}")