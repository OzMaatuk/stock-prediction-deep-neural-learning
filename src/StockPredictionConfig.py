import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@DeprecationWarning
class StockPredictionConfig:
    """Holds and validates configuration for stock prediction.

    Attributes:
        ticker (str): Stock ticker symbol (e.g., "GOOG").
        start_date (datetime): Data retrieval start date.
        end_date (datetime): Data retrieval end date.
        validation_date (datetime):  Train/validation split date.
        project_folder (str): Project directory path.
        epochs (int): Number of training epochs. Defaults to 100.
        time_steps (int):  LSTM time steps. Defaults to 60.
        batch_size (int): Training batch size. Defaults to 10. 
        csv_file (str): Path to the CSV data file.
        short_name (str): Short name of the stock (extracted from ticker).
        currency (str):  Currency of the stock (assumed to be USD for now).
    """

    def __init__(self, 
                 ticker: str, 
                 start_date: datetime, 
                 end_date: datetime, 
                 validation_date: datetime, 
                 project_folder: str, 
                 epochs: int = 100, 
                 time_steps: int = 60, 
                 batch_size: int = 10):

        # Input Validation
        if not all(isinstance(arg, str) and arg for arg in [ticker, project_folder]):
            raise TypeError("Ticker and project_folder must be non-empty strings.")
        if not all(isinstance(arg, datetime.datetime) for arg in [start_date, end_date, validation_date]):
            raise TypeError("Start_date, end_date, and validation_date must be datetime objects.")
        if not all(isinstance(arg, int) and arg > 0 for arg in [epochs, time_steps, batch_size]):
            raise ValueError("Epochs, time_steps, and batch_size must be positive integers.")
        if not os.path.exists(project_folder):
            raise ValueError(f"Project folder not found: {project_folder}") 
        if not start_date < end_date:
            raise ValueError("Start date must be before end date.")
        if not start_date < validation_date < end_date:
            raise ValueError("Validation date must be between start and end dates.")

        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.validation_date = validation_date
        self.project_folder = project_folder
        self.epochs = epochs
        self.time_steps = time_steps
        self.batch_size = batch_size

        self.csv_file = os.path.join(self.project_folder, 'data.csv')
        self.short_name = ticker  # You can add logic to extract a short name if needed
        self.currency = "USD"     # You might want to fetch the currency dynamically in the future