import pandas as pd
import yfinance as yf

# show data for different tickers
start = pd.to_datetime('2024-06-07')
end = pd.to_datetime('2024-06-27')


# stocks = ['^GSPC','GOOG','TSLA','HOOD','JMIA','TSEM']
stocks = ['GOOG']

for stock in stocks:
    data = yf.download(stock, start=start, end=end)
    # data = yf.download(stock, start=start, end=end, interval="1m")
    # print(data)
    data.to_csv(stock + '_unseen.csv')

