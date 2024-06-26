import datetime

class StockPredictionConfig:
    """
    Holds configuration parameters for the stock prediction project.
    """

    def __init__(self, 
                 ticker: str, 
                 start_date: datetime, 
                 end_date: datetime, 
                 validation_date: datetime, 
                 project_folder: str, 
                 epochs: int = 100, 
                 time_steps: int = 60, 
                 token: str = "", 
                 batch_size: int = 10):
        """
        Initializes the StockPredictionConfig object with configuration parameters.

        Args:
            ticker: The stock ticker symbol (e.g., "GOOG").
            start_date: The start date for data retrieval.
            end_date: The end date for data retrieval.
            validation_date: The date to split the data into training and validation sets.
            project_folder: The path to the project folder where results will be saved.
            epochs: The number of epochs for training the LSTM model.
            time_steps: The number of time steps to use for the LSTM model.
            token: A unique identifier for the project run.
            batch_size: The batch size for training the LSTM model.
        """

        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.validation_date = validation_date
        self.project_folder = project_folder
        self.epochs = epochs
        self.time_steps = time_steps
        self.token = token
        self.batch_size = batch_size

        self.CSV_FILE = f"{self.project_folder}data.csv"