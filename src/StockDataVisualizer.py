import os
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from src.StockPredictionConfig import StockPredictionConfig 
from src.StockDataProcessor import StockDataProcessor 

class StockDataVisualizer:
    """
    Handles plotting and visualization for stock prediction data.
    All methods are static.
    """

    @staticmethod
    def plot_histogram_data_split(training_data: pd.DataFrame, 
                                  test_data: pd.DataFrame, 
                                  validation_date: datetime, 
                                  project_folder: str, 
                                  short_name: str = "", 
                                  currency: str = "") -> None:
        """
        Plots the initial data split (training and validation data) and saves a histogram.

        Args:
            training_data: The training data DataFrame.
            test_data: The validation data DataFrame.
            validation_date: The date used to split the data.
            project_folder: The path to the project folder.
            short_name: The short name of the stock (e.g., "GOOG").
            currency: The currency of the stock (e.g., "USD").
        """

        print("plotting Data and Histogram")
        plt.figure(figsize=(12, 5))
        plt.plot(training_data.Close, color='green')
        plt.plot(test_data.Close, color='red')
        plt.ylabel('Price [' + currency + ']')
        plt.xlabel("Date")
        plt.legend(["Training Data", "Validation Data >= " + validation_date.strftime("%Y-%m-%d")])
        plt.title(short_name)
        plt.savefig(os.path.join(project_folder, 'price.png'))

        fig, ax = plt.subplots()
        training_data.hist(ax=ax)
        fig.savefig(os.path.join(project_folder, 'hist.png'))

        plt.pause(0.001)
        plt.show(block=True)  # Assuming you want to block the execution until the plot is closed

    @staticmethod
    def plot_loss(history, project_folder: str) -> None:
        """
        Plots the training loss and validation loss over epochs.

        Args:
            history: The training history object.
            project_folder: The path to the project folder.
        """

        print("plotting loss")
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss/Validation Loss')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(project_folder, 'loss.png'))
        plt.pause(0.001)
        plt.show(block=True)

    @staticmethod
    def plot_mse(history, project_folder: str) -> None:
        """
        Plots the training MSE and validation MSE over epochs.

        Args:
            history: The training history object.
            project_folder: The path to the project folder.
        """

        print("plotting MSE")
        plt.plot(history.history['MSE'], label='MSE')
        plt.plot(history.history['val_MSE'], label='val_MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('MSE/Validation MSE')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(project_folder, 'MSE.png'))
        plt.pause(0.001)
        plt.show(block=True)

    @staticmethod
    def plot_predictions(price_predicted: pd.DataFrame, 
                                 test_data: pd.DataFrame, 
                                 project_folder: str, 
                                 stock_ticker: str = "", 
                                 currency: str = "") -> None:
        """
        Plots the predicted stock prices against the actual prices or simulated prices.

        Args:
            price_predicted: The DataFrame containing the predicted prices.
            test_data: The original testing DataFrame or simulated data DataFrame.
            project_folder: The path to the project folder.
            stock_ticker: The stock ticker symbol.
            currency: The currency of the stock (e.g., "USD").
        """

        print("plotting predictions")
        plt.figure(figsize=(14, 5))

        print(price_predicted.columns)

        plt.plot(price_predicted['Predicted_Price'], color='red', label='Predicted [' + stock_ticker + '] price')
        plt.plot(test_data.Close, color='green', label='Actual [' + stock_ticker + '] price')
        plt.title('Predicted vs Actual Prices')

        plt.xlabel('Time')
        plt.ylabel('Price [' + currency + ']')
        plt.legend()
        plt.savefig(os.path.join(project_folder, 'prediction.png'))
        plt.pause(0.001)
        plt.show(block=True)

    @staticmethod
    def plot_future(combined_data: pd.DataFrame, 
                    project_folder: str, 
                    short_name: str = "",
                    currency: str = "") -> None:
        """
        Plots the predicted future stock prices against the simulated prices.

        Args:
            combined_data: The DataFrame containing both simulated and predicted prices.
            project_folder: The path to the project folder.
            stock_ticker: The stock ticker symbol.
            name: The short name of the stock (e.g., "GOOG").
            currency: The currency of the stock (e.g., "USD").
        """

        print("plotting future predictions")
        plt.figure(figsize=(14, 5))
        plt.plot(combined_data['Datetime'], combined_data.Close, color='green', label='Simulated [' + short_name + '] price')
        plt.plot(combined_data['Datetime'], combined_data['Predicted_Price'], color='red', label='Predicted [' + short_name + '] price')
        plt.xlabel('Time')
        plt.ylabel('Price [' + currency + ']')
        plt.legend()
        plt.title('Simulated vs Predicted Prices')
        plt.savefig(os.path.join(project_folder, 'future_comparison.png'))
        plt.pause(0.001)
        plt.show(block=True)

    @staticmethod
    def plot_results(stock_config: StockPredictionConfig, 
                        history, 
                        training_data: pd.DataFrame, 
                        test_data: pd.DataFrame, 
                        model, 
                        x_test) -> None:
        """
        Plots all results: data split, loss, MSE, and predictions.

        Args:
            stock_config: The configuration settings for the stock prediction project.
            history: The training history object.
            training_data: The training data DataFrame.
            test_data: The original testing DataFrame.
            data_processor: The data processor object.
            model: The trained LSTM model.
            x_test: The testing data for the LSTM model.
        """

        StockDataVisualizer.plot_histogram_data_split(training_data,
                                                      test_data,
                                                      stock_config.validation_date,
                                                      stock_config.project_folder)
        StockDataVisualizer.plot_loss(history, stock_config.project_folder)
        StockDataVisualizer.plot_mse(history, stock_config.project_folder)

        print("plotting prediction results")
        test_predictions_baseline = model.predict(x_test)
        test_predictions_baseline = StockDataProcessor.min_max.inverse_transform(test_predictions_baseline)
        test_predictions_baseline = pd.DataFrame(test_predictions_baseline, columns=['Predicted_Price'])
        test_predictions_baseline.to_csv(os.path.join(stock_config.project_folder, 'predictions.csv'))

        test_predictions_baseline = test_predictions_baseline.round(decimals=0)
        test_predictions_baseline.index = test_data.index
        StockDataVisualizer.plot_predictions(test_predictions_baseline, test_data, stock_config.project_folder)