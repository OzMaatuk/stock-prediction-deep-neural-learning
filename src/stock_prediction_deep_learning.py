# Copyright 2020-2024 Jordi Corbilla. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import pandas as pd

from src.stock_prediction_lstm import LongShortTermMemory
from src.stock_prediction_plotter import Plotter
from src.stock_prediction_readme_generator import ReadmeGenerator


def train_LSTM_network(data, stock, x_train, y_train, x_test, y_test, training_data, test_data):
    plotter = Plotter(True, stock.project_folder(), stock.ticker(), data.get_stock_currency(), stock.ticker())
    plotter.plot_histogram_data_split(training_data, test_data, stock.validation_date())

    lstm = LongShortTermMemory(stock.project_folder())
    model = lstm.create_model(x_train)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=lstm.get_defined_metrics())
    history = model.fit(x_train, y_train, epochs=stock.epochs(), batch_size=stock.batch_size(), validation_data=(x_test, y_test),
                        callbacks=[lstm.get_callback()])
    print("saving weights")
    model.save(os.path.join(stock.project_folder(), 'model_weights.keras'))

    plotter.plot_loss(history)
    plotter.plot_mse(history)

    print("display the content of the model")
    baseline_results = model.evaluate(x_test, y_test, verbose=2)
    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)
    print()

    print("plotting prediction results")
    test_predictions_baseline = model.predict(x_test)
    test_predictions_baseline = data.min_max.inverse_transform(test_predictions_baseline)
    test_predictions_baseline = pd.DataFrame(test_predictions_baseline)
    test_predictions_baseline.to_csv(os.path.join(stock.project_folder(), 'predictions.csv'))

    test_predictions_baseline.rename(columns={0: stock.ticker() + '_predicted'}, inplace=True)
    test_predictions_baseline = test_predictions_baseline.round(decimals=0)
    test_predictions_baseline.index = test_data.index
    plotter.project_plot_predictions(test_predictions_baseline, test_data)

    # generator = ReadmeGenerator(stock.project_folder(), stock.ticker())
    # generator.write()

    print("prediction is finished")