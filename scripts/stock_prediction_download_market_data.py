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

import pandas as pd
import yfinance as yf
import datetime

# show data for different tickers
start = pd.to_datetime('2024-05-24')
end = pd.to_datetime('2024-05-31')
stock = ['^GSPC']
data = yf.download(stock, start=start, end=end, interval="1m")
# print(data)
data.to_csv('^GSPC_1_min.csv')


# stock = ['GOOG']
# data = yf.download(stock, start=start, end=end, interval="1m")
# # print(data)
# data.to_csv('GOOG_1_min.csv')


# stock = ['TSLA']
# data = yf.download(stock, start=start, end=datetime.date.today())
# # print(data)
# data.to_csv('TSLA_1_min.csv')


# stock = ['HOOD']
# data = yf.download(stock, start=start, end=datetime.date.today())
# # print(data)
# data.to_csv('HOOD_1_min.csv')



# stock = ['JMIA']
# data = yf.download(stock, start=start, end=datetime.date.today())
# # print(data)
# data.to_csv('JMIA_1_min.csv')



# stock = ['TSEM']
# data = yf.download(stock, start=start, end=datetime.date.today())
# # print(data)
# data.to_csv('TSEM_1_min.csv')


