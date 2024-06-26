import os
import tensorflow as tf
import pandas as pd


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


    @staticmethod
    def Train(stock_config: StockPredictionConfig, x_train, y_train, x_test, y_test, training_data, test_data) -> None:
        """
        Trains the LSTM network, saves the model weights, and evaluates its performance.

        Args:
            stock_config: The configuration settings for the stock prediction project.
            data_processor: The data processor for loading and transforming stock data.
        """

        # Load and transform data
        (x_train, y_train), (x_test, y_test), (training_data, test_data) = \
            StockDataProcessor.load_csv_transform_to_numpy(
                stock_config.time_steps,
                stock_config.CSV_FILE,
                stock_config.validation_date
            )

        # Create and train LSTM model
        lstm = LSTMModel(stock_config.project_folder)
        model = lstm.create()
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=lstm.get_metrics())
        history = model.fit(x_train, y_train,
                            epochs=stock_config.epochs,
                            batch_size=stock_config.batch_size,
                            validation_data=(x_test, y_test),
                            callbacks=lstm.get_callbacks())

        # Save model weights
        print("saving weights")
        model.save(os.path.join(stock_config.project_folder, 'model_weights.keras'))

        # Evaluate model performance
        lstm.evaluate(model, x_test, y_test)

        # Plot results
        StockDataVisualizer.plot_results(stock_config, history, training_data, test_data, model, x_test)

        print("prediction is finished")


    @staticmethod
    def evaluate(model, x_test, y_test):
        """
        Evaluates the trained model on the test data.

        Args:
            model: The trained LSTM model.
            x_test: The testing data for the LSTM model.
            y_test: The target values for the testing data.
        """

        print("display the content of the model")
        baseline_results = model.evaluate(x_test, y_test, verbose=2)
        for name, value in zip(model.metrics_names, baseline_results):
            print(name, ': ', value)
        print()



    @staticmethod
    def Infer(start_date, end_date, latest_close_price, work_dir, time_steps):
        x_test, y_test, test_data = StockDataProcessor.generate_future_data(time_steps, start_date, end_date, latest_close_price)

        # Check if the future data is not empty
        if x_test.shape[0] > 0:
            # load the weights from our best model
            model = tf.keras.models.load_model(os.path.join(work_dir, 'model_weights.keras'))
            model.summary()

            # perform a prediction
            test_predictions_baseline = model.predict(x_test)
            test_predictions_baseline = StockDataProcessor.min_max.inverse_transform(test_predictions_baseline)
            test_predictions_baseline = pd.DataFrame(test_predictions_baseline, columns=['Predicted_Price'])

            # Combine the predicted values with dates from the test data
            if (StockDataProcessor.is_day_interval(start_date, end_date)):
                predicted_dates = pd.date_range(start=test_data.index[0], periods=len(test_predictions_baseline))
            else:
                predicted_dates = pd.date_range(start=test_data.index[0], periods=len(test_predictions_baseline), freq="1min")
            test_predictions_baseline['Datetime'] = predicted_dates
            
            # Reset the index for proper concatenation
            test_data.reset_index(inplace=True)
            
            # Concatenate the test_data and predicted data
            combined_data = pd.concat([test_data, test_predictions_baseline], ignore_index=True)
            
            # Plotting predictions
            StockDataVisualizer.plot_future(combined_data, work_dir)
        else:
            print("Error: Future data is empty.")