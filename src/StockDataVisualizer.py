import os
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import logging

from src.StockPredictionConfig import StockPredictionConfig 
from src.StockDataProcessor import StockDataProcessor 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StockDataVisualizer:
    """Handles plotting and visualization of stock prediction data.

    All methods are static.
    """

    @staticmethod
    def plot_histogram_data_split(training_data: pd.DataFrame, 
                                  test_data: pd.DataFrame, 
                                  validation_date: datetime, 
                                  project_folder: str, 
                                  short_name: str = "", 
                                  currency: str = "") -> None:
        """Plots initial data split and saves a histogram.

        Args:
            training_data (pd.DataFrame): Training data.
            test_data (pd.DataFrame): Testing/validation data.
            validation_date (datetime): Split date.
            project_folder (str): Directory to save plots.
            short_name (str, optional): Stock short name. Defaults to "".
            currency (str, optional): Currency symbol. Defaults to "".
        """
        try:
            logging.info("Plotting data split and histogram...")
            plt.figure(figsize=(12, 5))
            plt.plot(training_data.index, training_data.Close, color='green')
            plt.plot(test_data.index, test_data.Close, color='red')
            plt.ylabel(f'Price [{currency}]')
            plt.xlabel("Date")
            plt.legend(["Training Data", f"Validation Data >= {validation_date.strftime('%Y-%m-%d')}"], loc='upper left')
            plt.title(short_name)
            plt.tight_layout()  # Adjust layout to prevent labels from overlapping
            plt.savefig(os.path.join(project_folder, 'price.png'))

            fig, ax = plt.subplots()
            training_data.hist(ax=ax)
            fig.savefig(os.path.join(project_folder, 'hist.png'))

            # Display plots 
            plt.show() 

        except Exception as e:
            logging.error(f"Error plotting data split: {e}")

    @staticmethod
    def plot_loss(history, project_folder: str) -> None:
        """Plots training and validation loss over epochs.

        Args:
            history: The training history object.
            project_folder (str): Directory to save plot.
        """
        try:
            logging.info("Plotting loss...")
            plt.plot(history.history['loss'], label='loss')
            plt.plot(history.history['val_loss'], label='val_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss/Validation Loss')
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(project_folder, 'loss.png'))
            plt.show()  

        except Exception as e:
            logging.error(f"Error plotting loss: {e}")

    @staticmethod
    def plot_mse(history, project_folder: str) -> None:
        """Plots training and validation MSE over epochs.

        Args:
            history: The training history object.
            project_folder (str): Directory to save plot.
        """
        try:
            logging.info("Plotting MSE...")
            plt.plot(history.history['MSE'], label='MSE')
            plt.plot(history.history['val_MSE'], label='val_MSE')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.title('MSE/Validation MSE')
            plt.legend(loc='upper right')
            plt.tight_layout() 
            plt.savefig(os.path.join(project_folder, 'MSE.png'))
            plt.show() 

        except Exception as e:
            logging.error(f"Error plotting MSE: {e}")

    @staticmethod
    def plot_predictions(price_predicted: pd.DataFrame, 
                                 test_data: pd.DataFrame, 
                                 project_folder: str, 
                                 stock_ticker: str = "", 
                                 currency: str = "") -> None:
        """Plots predicted vs. actual/simulated stock prices.

        Args:
            price_predicted (pd.DataFrame): DataFrame of predicted prices.
            test_data (pd.DataFrame):  Original or simulated test data.
            project_folder (str): Directory to save plot.
            stock_ticker (str, optional): Stock ticker symbol. Defaults to "".
            currency (str, optional):  Currency symbol. Defaults to "".
        """
        try:
            logging.info("Plotting predictions...")
            plt.figure(figsize=(14, 5))
            plt.plot(price_predicted.index, price_predicted['Predicted_Price'], color='red', 
                     label=f'Predicted [{stock_ticker}] price')
            plt.plot(test_data.index, test_data.Close, color='green', label=f'Actual [{stock_ticker}] price')
            plt.title('Predicted vs Actual Prices')
            plt.xlabel('Time')
            plt.ylabel(f'Price [{currency}]')
            plt.legend()
            plt.tight_layout() 
            plt.savefig(os.path.join(project_folder, 'prediction.png'))
            plt.show()  

        except Exception as e:
            logging.error(f"Error plotting predictions: {e}")

    @staticmethod
    def plot_future(test_data: pd.DataFrame,
                    predicted_data: pd.DataFrame, 
                    project_folder: str, 
                    short_name: str = "",
                    currency: str = "") -> None:
        """Plots predicted future stock prices vs. simulated prices.

        Args:
            combined_data (pd.DataFrame):  Combined simulated and predicted data.
            project_folder (str): Directory to save plot.
            short_name (str, optional): Stock short name. Defaults to "".
            currency (str, optional): Currency symbol. Defaults to "".
        """
        try:
            logging.info("Plotting future predictions...")
            plt.figure(figsize=(14, 5))
            plt.plot(test_data.index, test_data['Close'], color='green', label=f'Simulated {short_name} price')
            plt.plot(predicted_data.index, predicted_data['Predicted'], color='red', label=f'Predicted {short_name} price')
            plt.xlabel('Time')
            plt.ylabel(f'Price {currency}')
            plt.legend()
            plt.title('Simulated vs Predicted Prices')
            plt.tight_layout()  
            plt.savefig(os.path.join(project_folder, 'future_comparison.png'))
            plt.show() 

        except Exception as e:
            logging.error(f"Error plotting future predictions: {e}")

    @staticmethod
    def plot_results(stock_config: StockPredictionConfig, 
                        history, 
                        training_data: pd.DataFrame, 
                        test_data: pd.DataFrame, 
                        model, 
                        x_test) -> None:
        """Plots all results (data split, loss, MSE, predictions).

        Args:
            stock_config (StockPredictionConfig): Configuration settings.
            history:  The training history object.
            training_data (pd.DataFrame):  Training data.
            test_data (pd.DataFrame): Testing data.
            model: The trained LSTM model.
            x_test:  Testing data for predictions.
        """
        try:
            StockDataVisualizer.plot_histogram_data_split(
                training_data, test_data, stock_config.validation_date, 
                stock_config.project_folder, stock_config.short_name,
                stock_config.currency
            )
            StockDataVisualizer.plot_loss(history, stock_config.project_folder)
            StockDataVisualizer.plot_mse(history, stock_config.project_folder)

            logging.info("Plotting prediction results...")
            test_predictions_baseline = model.predict(x_test)
            test_predictions_baseline = StockDataProcessor.min_max.inverse_transform(test_predictions_baseline)
            test_predictions_baseline = pd.DataFrame(test_predictions_baseline, columns=['Predicted_Price'])
            test_predictions_baseline.to_csv(os.path.join(stock_config.project_folder, 'predictions.csv'))

            test_predictions_baseline = test_predictions_baseline.round(decimals=0)
            test_predictions_baseline.index = test_data.index
            StockDataVisualizer.plot_predictions(
                test_predictions_baseline, test_data, stock_config.project_folder, 
                stock_config.ticker, stock_config.currency 
            ) 

        except Exception as e:
            logging.error(f"Error in plot_results: {e}")