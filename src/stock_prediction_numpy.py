import os

import numpy as np
from datetime import timedelta
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf


class DataClass:
    min_max = MinMaxScaler(feature_range=(0, 1))

    def __data_verification(self, train):
        print('mean:', train.mean(axis=0))
        print('max', train.max())
        print('min', train.min())
        print('Std dev:', train.std(axis=0))

    def get_stock_currency(self):
        # return self._sec.info['currency']
        return "USD"

    def download_transform_to_numpy(self, ticker, time_steps, project_folder, start_date, end_date, validation_date):
        data = yf.download([ticker], start_date, end_date)[['Close']]
        data = data.reset_index()
        data.to_csv(os.path.join(project_folder, 'data.csv'))
        #print(data)
        return self.transform_numpy(data, time_steps, validation_date)

    def load_csv_transform_to_numpy(self, time_steps, csv_path, validation_date):
        data = pd.read_csv(csv_path, index_col=0)
        # data = data.reset_index()
        return self.transform_numpy(data, time_steps, validation_date)

    def transform_numpy(self, data, time_steps, validation_date):
        date_col = pd.to_datetime(data['Date'])
        data['Date'] = date_col
        training_data = data[date_col < validation_date].copy()
        test_data = data[date_col >= validation_date].copy()
        training_data = training_data.set_index('Date')
        # Set the data frame index using column Date
        test_data = test_data.set_index('Date')
        #print(test_data)

        train_scaled = self.min_max.fit_transform(training_data)
        self.__data_verification(train_scaled)

        # Training Data Transformation
        x_train = []
        y_train = []
        for i in range(time_steps, train_scaled.shape[0]):
            x_train.append(train_scaled[i - time_steps:i])
            y_train.append(train_scaled[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        total_data = pd.concat((training_data, test_data), axis=0)
        inputs = total_data[len(total_data) - len(test_data) - time_steps:]
        test_scaled = self.min_max.fit_transform(inputs)

        # Testing Data Transformation
        x_test = []
        y_test = []
        for i in range(time_steps, test_scaled.shape[0]):
            x_test.append(test_scaled[i - time_steps:i])
            y_test.append(test_scaled[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return (x_train, y_train), (x_test, y_test), (training_data, test_data)

    def __date_range(self, start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    def negative_positive_random(self):
        return 1 if random.random() < 0.5 else -1

    def pseudo_random(self):
        return random.uniform(0.01, 0.03)

    def generate_future_data(self, time_steps, start_date, end_date, latest_close_price):
        x_future = []
        y_future = []

        # We need to provide a randomisation algorithm for the close price
        # This is my own implementation and it will provide a variation of the
        # close price for a +-1-3% of the original value, when the value wants to go below
        # zero, it will be forced to go up.

        for single_date in self.__date_range(start_date, end_date):
            x_future.append(single_date)
            direction = self.negative_positive_random()
            random_slope = direction * (self.pseudo_random())
            #print(random_slope)
            latest_close_price = latest_close_price + (latest_close_price * random_slope)
            #print(original_price)
            if latest_close_price < 0:
                latest_close_price = 0
            y_future.append(latest_close_price)

        test_data = pd.DataFrame({'Date': x_future, 'Close': y_future})
        test_data = test_data.set_index('Date')

        test_scaled = self.min_max.fit_transform(test_data)
        x_test = []
        y_test = []
        #print(test_scaled.shape[0])
        for i in range(time_steps, test_scaled.shape[0]):
            x_test.append(test_scaled[i - time_steps:i])
            y_test.append(test_scaled[i, 0])
            #print(i - time_steps)

        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return x_test, y_test, test_data



