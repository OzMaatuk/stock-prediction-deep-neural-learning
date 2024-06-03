import pandas as pd

# Adding index column
CSV_FILE = 'data/min/TSLA.csv'
data = pd.read_csv(CSV_FILE)
data.reset_index(inplace=True, drop=True)
data.to_csv(CSV_FILE)