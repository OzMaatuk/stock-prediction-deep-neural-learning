import os
from datetime import datetime, timedelta
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import numpy as np

class StockDataProcessor:
    """
    Handles data loading, preprocessing, transformation, and generation of future data for stock prediction.
    All methods are static.
    """

    min_max = MinMaxScaler(feature_range=(0, 1))

    @staticmethod
    def __data_verification(train: pd.DataFrame):
        """
        Prints basic statistics about the data (mean, max, min, standard deviation).

        Args:
            train: The training data DataFrame.
        """

        print('mean:', train.mean(axis=0))
        print('max:', train.max())
        print('min:', train.min())
        print('Std dev:', train.std(axis=0))

    @staticmethod
    def download_transform_to_numpy(ticker: str, 
                                  time_steps: int, 
                                  project_folder: str, 
                                  start_date: datetime, 
                                  end_date: datetime, 
                                  validation_date: datetime) -> tuple:
        """
        Downloads stock data from Yahoo Finance, saves it to CSV, and transforms
        it into NumPy arrays for training and testing.

        Args:
            ticker: The stock ticker symbol.
            time_steps: The number of time steps for the LSTM model.
            project_folder: The path to the project folder.
            start_date: The start date for data retrieval.
            end_date: The end date for data retrieval.
            validation_date: The date to split the data into training and validation sets.

        Returns:
            tuple: A tuple containing:
                * x_train: The training data for the LSTM model.
                * y_train: The target values for the training data.
                * x_test: The testing data for the LSTM model.
                * y_test: The target values for the testing data.
                * training_data: The original training DataFrame.
                * test_data: The original testing DataFrame.
        """

        data = yf.download([ticker], start_date, end_date)[['Close']]
        data = data.reset_index()
        data.to_csv(os.path.join(project_folder, 'data.csv'))

        return StockDataProcessor.transform_numpy(data, time_steps, validation_date)

    @staticmethod
    def load_csv_transform_to_numpy(time_steps: int, 
                                    csv_path: str, 
                                    validation_date: datetime) -> tuple:
        """
        Loads stock data from a CSV file, transforms it into NumPy arrays for 
        training and testing.

        Args:
            time_steps: The number of time steps for the LSTM model.
            csv_path: The path to the CSV file.
            validation_date: The date to split the data into training and validation sets.

        Returns:
            tuple: A tuple containing:
                * x_train: The training data for the LSTM model.
                * y_train: The target values for the training data.
                * x_test: The testing data for the LSTM model.
                * y_test: The target values for the testing data.
                * training_data: The original training DataFrame.
                * test_data: The original testing DataFrame.
        """

        data = pd.read_csv(csv_path, index_col=0)
        data.drop(columns=["Open","High","Low","Adj Close","Volume"], inplace=True)
        return StockDataProcessor.transform_numpy(data, time_steps, validation_date)

    @staticmethod
    def transform_numpy(data: pd.DataFrame, 
                         time_steps: int, 
                         validation_date: datetime) -> tuple:
        """
        Transforms the stock data into NumPy arrays for training and testing.

        Args:
            data: The stock data DataFrame.
            time_steps: The number of time steps for the LSTM model.
            validation_date: The date to split the data into training and validation sets.

        Returns:
            tuple: A tuple containing:
                * x_train: The training data for the LSTM model.
                * y_train: The target values for the training data.
                * x_test: The testing data for the LSTM model.
                * y_test: The target values for the testing data.
                * training_data: The original training DataFrame.
                * test_data: The original testing DataFrame.
        """

        date_col = pd.to_datetime(data['Datetime'])
        data['Datetime'] = date_col
        training_data = data[date_col < validation_date].copy()
        test_data = data[date_col >= validation_date].copy()
        training_data = training_data.set_index('Datetime')
        test_data = test_data.set_index('Datetime')

        train_scaled = StockDataProcessor.min_max.fit_transform(training_data)
        StockDataProcessor.__data_verification(train_scaled)

        # Training Data Transformation
        x_train, y_train = StockDataProcessor._create_sequences(train_scaled, time_steps)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        total_data = pd.concat((training_data, test_data), axis=0)
        inputs = total_data[len(total_data) - len(test_data) - time_steps:]
        test_scaled = StockDataProcessor.min_max.fit_transform(inputs)

        # Testing Data Transformation
        x_test, y_test = StockDataProcessor._create_sequences(test_scaled, time_steps)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        return (x_train, y_train), (x_test, y_test), (training_data, test_data)

    @staticmethod
    def _create_sequences(data: np.ndarray, time_steps: int) -> tuple:
        """
        Creates sequences for the LSTM model from the provided data.

        Args:
            data: The preprocessed data array.
            time_steps: The number of time steps for each sequence.

        Returns:
            tuple: A tuple containing:
                * x: The input sequences for the LSTM model.
                * y: The target values for the sequences.
        """

        x = []
        y = []
        for i in range(time_steps, data.shape[0]):
            x.append(data[i - time_steps:i])
            y.append(data[i, 0])

        return np.array(x), np.array(y)

    @staticmethod
    def calc_date_range(start_date: datetime, 
                        end_date: datetime) -> list[datetime]:
        """
        Calculates a list of dates between the start and end dates.

        Args:
            start_date: The start date.
            end_date: The end date.

        Returns:
            list[datetime]: A list of dates between the start and end dates.
        """

        is_start_day_is_full_day = True if (start_date.hour == start_date.minute == 0) else False
        is_end_day_is_full_day = True if (end_date.hour == end_date.minute == 0) else False
        is_day_interval = is_start_day_is_full_day and is_end_day_is_full_day
        date_range = []
        if (not is_day_interval):
            my_range = (end_date - start_date).total_seconds() / 60
            for n in range(int(my_range)):
                date_range.append(start_date + timedelta(minutes=n))
        else:
            my_range = (end_date - start_date).days
            for n in range(int(my_range)):
                date_range.append(start_date + timedelta(days=n))
        return date_range

    @staticmethod
    def negative_positive_random() -> int:
        """
        Returns a random integer, either 1 or -1.

        Returns:
            int: Either 1 or -1.
        """

        return 1 if random.random() < 0.5 else -1

    @staticmethod
    def pseudo_random() -> float:
        """
        Returns a random float between 0.01 and 0.03.

        Returns:
            float: A random float between 0.01 and 0.03.
        """

        return random.uniform(0.01, 0.03)

    @staticmethod
    def generate_future_data(time_steps: int, 
                             start_date: datetime, 
                             end_date: datetime, 
                             latest_close_price: float) -> tuple:
        """
        Generates simulated future stock data based on the latest close price.

        Args:
            time_steps: The number of time steps for the LSTM model.
            start_date: The start date for the simulated data.
            end_date: The end date for the simulated data.
            latest_close_price: The latest closing price of the stock.

        Returns:
            tuple: A tuple containing:
                * x_test: The input sequences for the future data.
                * y_test: The target values for the future data.
                * test_data: The DataFrame containing the simulated future data.
        """

        x_future = []
        y_future = []

        for single_date in StockDataProcessor.calc_date_range(start_date, end_date):
            x_future.append(single_date)
            direction = StockDataProcessor.negative_positive_random()
            random_slope = direction * (StockDataProcessor.pseudo_random())
            latest_close_price = latest_close_price + (latest_close_price * random_slope)
            if latest_close_price < 0:
                latest_close_price = 0
            y_future.append(latest_close_price)

        test_data = pd.DataFrame({'Datetime': x_future, 'Close': y_future})
        test_data = test_data.set_index('Datetime')

        test_scaled = StockDataProcessor.min_max.fit_transform(test_data)
        x_test, y_test = StockDataProcessor._create_sequences(test_scaled, time_steps)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        return x_test, y_test, test_data