import os
from datetime import datetime, timedelta
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StockDataProcessor:
    """Handles data loading, preprocessing, transformation, and generation of future data.

    All methods are static.

    Attributes:
        min_max (MinMaxScaler): A MinMaxScaler instance for data normalization.
    """

    min_max = MinMaxScaler(feature_range=(0, 1))

    @staticmethod
    def __data_verification(train: pd.DataFrame):
        """Prints basic data statistics (mean, max, min, std dev).

        Args:
            train (pd.DataFrame): The training data DataFrame.
        """

        logging.info("Data Statistics:")
        logging.info(f"Mean:\n{train.mean(axis=0)}")
        logging.info(f"Max:\n{train.max()}")
        logging.info(f"Min:\n{train.min()}")
        logging.info(f"Std Dev:\n{train.std(axis=0)}")

    @staticmethod
    def download_transform_to_numpy(ticker: str, 
                                  time_steps: int, 
                                  project_folder: str, 
                                  start_date: datetime, 
                                  end_date: datetime, 
                                  validation_date: datetime) -> tuple:
        """Downloads, saves, and transforms stock data for training/testing.

        Args:
            ticker (str): The stock ticker symbol.
            time_steps (int): LSTM time steps.
            project_folder (str): Project directory path.
            start_date (datetime): Data retrieval start date.
            end_date (datetime): Data retrieval end date.
            validation_date (datetime): Train/validation split date.

        Returns:
            tuple: (x_train, y_train, x_test, y_test, training_data, test_data)

        Raises:
            ValueError: If invalid dates or empty data from Yahoo Finance.
            Exception: For other errors during download or processing.
        """
        try:
            if start_date >= end_date:
                raise ValueError("Start date must be before end date.")
            if validation_date <= start_date or validation_date >= end_date:
                raise ValueError("Validation date must be between start and end dates.")

            data = yf.download([ticker], start=start_date, end=end_date)[['Close']]
            
            if data.empty:
                raise ValueError(f"No data found for {ticker} from {start_date} to {end_date}")
            
            data = data.reset_index()
            data.to_csv(os.path.join(project_folder, 'data.csv'))

            return StockDataProcessor.transform_numpy(data, time_steps, validation_date)

        except ValueError as ve:
            logging.error(f"Data validation error: {ve}")
            raise 
        except Exception as e:
            logging.error(f"An error occurred during download and processing: {e}")
            raise

    @staticmethod
    def load_csv_transform_to_numpy(time_steps: int, 
                                    csv_path: str, 
                                    validation_date: datetime) -> tuple:
        """Loads and transforms stock data from CSV for training/testing.

        Args:
            time_steps (int): LSTM time steps.
            csv_path (str): Path to the CSV file.
            validation_date (datetime): Train/validation split date.

        Returns:
            tuple: (x_train, y_train, x_test, y_test, training_data, test_data)

        Raises:
            FileNotFoundError: If the CSV file is not found.
            ValueError: If the CSV data is empty or invalid.
            Exception: For other errors during loading or processing.
        """
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

            data = pd.read_csv(csv_path, index_col=0)
            
            if data.empty:
                raise ValueError("CSV file is empty.")
            
            if 'Datetime' not in data.columns or 'Close' not in data.columns:
                raise ValueError("CSV must have 'Datetime' and 'Close' columns.")
            
            data.drop(columns=["Open", "High", "Low", "Adj Close", "Volume"], errors='ignore', inplace=True)
            
            return StockDataProcessor.transform_numpy(data, time_steps, validation_date)

        except (FileNotFoundError, ValueError) as e:
            logging.error(e)
            raise 
        except Exception as e:
            logging.error(f"An error occurred during CSV loading and processing: {e}")
            raise 

    @staticmethod
    def transform_numpy(data: pd.DataFrame, 
                         time_steps: int, 
                         validation_date: datetime) -> tuple:
        """Transforms stock data into NumPy arrays for training/testing.

        Args:
            data (pd.DataFrame): The stock data DataFrame.
            time_steps (int): LSTM time steps.
            validation_date (datetime): Train/validation split date.

        Returns:
            tuple: (x_train, y_train, x_test, y_test, training_data, test_data)

        Raises:
            TypeError: If input data is not a pandas DataFrame.
            ValueError: If data is missing required columns.
            Exception: For other errors during transformation. 
        """
        try:
            if not isinstance(data, pd.DataFrame):
                raise TypeError("Input data must be a pandas DataFrame.")
            
            required_cols = ['Datetime', 'Close']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"DataFrame must have columns: {required_cols}")

            data['Datetime'] = pd.to_datetime(data['Datetime'])
            date_col = data['Datetime']
            training_data = data[date_col < validation_date].copy()
            test_data = data[date_col >= validation_date].copy()
            training_data = training_data.set_index('Datetime')
            test_data = test_data.set_index('Datetime')

            train_scaled = StockDataProcessor.min_max.fit_transform(training_data[['Close']])
            StockDataProcessor.__data_verification(training_data[['Close']]) 

            # Training Data Transformation
            x_train, y_train = StockDataProcessor._create_sequences(train_scaled, time_steps)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            total_data = pd.concat((training_data, test_data), axis=0)
            inputs = total_data[len(total_data) - len(test_data) - time_steps:]
            test_scaled = StockDataProcessor.min_max.fit_transform(inputs[['Close']])

            # Testing Data Transformation
            x_test, y_test = StockDataProcessor._create_sequences(test_scaled, time_steps)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            return (x_train, y_train), (x_test, y_test), (training_data, test_data)

        except (TypeError, ValueError) as e:
            logging.error(e)
            raise 
        except Exception as e:
            logging.error(f"An error occurred during data transformation: {e}")
            raise

    @staticmethod
    def _create_sequences(data: np.ndarray, time_steps: int) -> tuple:
        """Creates input sequences for the LSTM model.

        Args:
            data (np.ndarray): The preprocessed data array.
            time_steps (int):  Number of time steps per sequence.

        Returns:
            tuple: (x, y) - NumPy arrays of input sequences and target values.

        Raises:
            ValueError: If time_steps is not a positive integer.
        """
        if time_steps <= 0 or not isinstance(time_steps, int):
            raise ValueError("Time steps must be a positive integer.")

        x = []
        y = []
        for i in range(time_steps, data.shape[0]):
            x.append(data[i - time_steps:i])
            y.append(data[i, 0])

        return np.array(x), np.array(y)

    @staticmethod
    def is_day_interval(start_date: datetime, end_date: datetime) -> bool:
        """Checks if the time interval between two dates is in full days.

        Args:
            start_date (datetime): Starting datetime object.
            end_date (datetime): Ending datetime object.

        Returns:
            bool: True if interval is in full days, False otherwise.
        """

        is_start_day_is_full_day = True if (start_date.hour == start_date.minute == 0) else False
        is_end_day_is_full_day = True if (end_date.hour == end_date.minute == 0) else False
        return is_start_day_is_full_day and is_end_day_is_full_day

    @staticmethod
    def calc_date_range(start_date: datetime, end_date: datetime) -> list[datetime]:
        """Generates a list of dates between start and end dates.

        Args:
            start_date (datetime): The start date.
            end_date (datetime): The end date.

        Returns:
            list[datetime]: A list of datetime objects.

        Raises:
            ValueError: If start_date is greater than or equal to end_date.
        """
        if start_date >= end_date:
            raise ValueError("Start date must be before end date.")

        date_range = []
        if StockDataProcessor.is_day_interval(start_date, end_date):
            current_date = start_date
            while current_date < end_date:
                date_range.append(current_date)
                current_date += timedelta(days=1) 
        else:
            current_date = start_date
            while current_date < end_date:
                date_range.append(current_date)
                current_date += timedelta(minutes=1)  

        return date_range

    @staticmethod
    def negative_positive_random() -> int:
        """Returns a random integer, either 1 or -1."""
        return 1 if random.random() < 0.5 else -1

    @staticmethod
    def pseudo_random() -> float:
        """Returns a random float between 0.01 and 0.03."""
        return random.uniform(0.01, 0.03)

    @staticmethod
    def generate_future_data(time_steps: int, 
                             start_date: datetime, 
                             end_date: datetime, 
                             latest_close_price: float) -> tuple:
        """Generates simulated future stock data.

        Args:
            time_steps (int): Number of time steps for the LSTM model.
            start_date (datetime): Start date for simulation.
            end_date (datetime): End date for simulation.
            latest_close_price (float): Last known closing price.

        Returns:
            tuple: (x_test, y_test, test_data) - Simulated data for testing.

        Raises:
            ValueError: If start_date is greater than or equal to end_date.
        """
        if start_date >= end_date:
            raise ValueError("Start date must be before end date.")

        x_future = []
        y_future = []

        for single_date in StockDataProcessor.calc_date_range(start_date, end_date):
            x_future.append(single_date)
            direction = StockDataProcessor.negative_positive_random()
            random_slope = direction * (StockDataProcessor.pseudo_random())
            latest_close_price = latest_close_price + (latest_close_price * random_slope)
            latest_close_price = max(0, latest_close_price)  # Ensure price doesn't go negative
            y_future.append(latest_close_price)

        test_data = pd.DataFrame({'Datetime': x_future, 'Close': y_future})
        test_data = test_data.set_index('Datetime')

        test_scaled = StockDataProcessor.min_max.fit_transform(test_data[['Close']])
        x_test, y_test = StockDataProcessor._create_sequences(test_scaled, time_steps)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        return x_test, y_test, test_data