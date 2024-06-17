import os
import tensorflow as tf
import pandas as pd

from src.LongShortTermMemory import LongShortTermMemory
from src.StockDataVisualizer import StockDataVisualizer
from src.StockPredictionConfig import StockPredictionConfig
from src.StockDataProcessor import StockDataProcessor

@staticmethod
def TrainLstmNetwork(stock_config: StockPredictionConfig, x_train, y_train, x_test, y_test, training_data, test_data) -> None:
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
    lstm = LongShortTermMemory(stock_config.project_folder)
    model = lstm.create_model(x_train)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=lstm.get_defined_metrics())
    history = model.fit(x_train, y_train,
                        epochs=stock_config.epochs,
                        batch_size=stock_config.batch_size,
                        validation_data=(x_test, y_test),
                        callbacks=[lstm.get_callback()])

    # Save model weights
    print("saving weights")
    model.save(os.path.join(stock_config.project_folder, 'model_weights.keras'))

    # Evaluate model performance
    evaluate_model(model, x_test, y_test)

    # Plot results
    StockDataVisualizer.plot_results(stock_config, history, training_data, test_data, model, x_test)

    print("prediction is finished")

def evaluate_model(model, x_test, y_test):
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
def InferLstmModel(start_date, end_date, latest_close_price, work_dir, time_steps, ticker, currency):
    
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
        predicted_dates = pd.date_range(start=test_data.index[0], periods=len(test_predictions_baseline))
        # predicted_dates = pd.date_range(start=test_data.index[0], periods=len(test_predictions_baseline), freq="1min")
        test_predictions_baseline['Datetime'] = predicted_dates
        
        # Reset the index for proper concatenation
        test_data.reset_index(inplace=True)
        
        # Concatenate the test_data and predicted data
        combined_data = pd.concat([test_data, test_predictions_baseline], ignore_index=True)
        
        # Plotting predictions
        StockDataVisualizer.plot_future(combined_data, work_dir, ticker, currency)
    else:
        print("Error: Future data is empty.")