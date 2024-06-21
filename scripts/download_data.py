import pandas as pd
import yfinance as yf
import datetime

# show data for different tickers
start = pd.to_datetime('2024-06-14')
end = pd.to_datetime('2024-06-21')


stocks = ['^GSPC','GOOG','TSLA','HOOD','JMIA','TSEM']

for stock in stocks:
    # data = yf.download(stock, start=start, end=datetime.date.today())
    data = yf.download(stock, start=start, end=end, interval="1m")
    # print(data)
    data.to_csv(stock + '_4.csv')

