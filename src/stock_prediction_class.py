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


class StockClass:
    def __init__(self, ticker, start_date, end_date, validation_date, project_folder, epochs, time_steps, token, batch_size):
        self.ticker = ticker
        self.start_date = start_date
        self.validation_date = validation_date
        self.project_folder = project_folder
        self.epochs = epochs
        self.time_steps = time_steps
        self.token = token
        self.batch_size = batch_size
        self.end_date = end_date