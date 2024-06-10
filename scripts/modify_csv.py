import pandas as pd
import os

# Adding index column
# CSV_FILE = 'data/min/TSLA.csv'
# data = pd.read_csv(CSV_FILE)
# data.reset_index(inplace=True, drop=True)
# data.to_csv(CSV_FILE)

# remove index column
# CSV_FILE = 'data/backup/min/GOOG_1.csv'
# data = pd.read_csv(CSV_FILE, index_col=0)
# data.to_csv(CSV_FILE, index=False)


# get all files in directory
DIR_PATH = "data/backup/min/"
entries = os.listdir(DIR_PATH)
files = [f for f in entries if os.path.isfile(os.path.join(DIR_PATH, f))]

# # remove column
# for file_path in files:
#     relative_file_path = DIR_PATH + file_path
#     print(relative_file_path)    
#     col_name = "Unnamed: 0.1"
#     data = pd.read_csv(relative_file_path)
#     cols = data.columns
#     if col_name in cols:
#         data.drop(columns=[col_name] ,inplace=True)
#         data.to_csv(relative_file_path)


# remove index column from csv
# for file_path in files:
#     relative_file_path = DIR_PATH + file_path
#     print(relative_file_path)
#     if (file_path.endswith("_1.csv")):
#         df = pd.read_csv(relative_file_path, index_col=0)
#         df.to_csv(relative_file_path, index=False)




# files = ["GOOG_1.csv", "GSPC_1.csv"]




# combine csv files
# for file_path in files:
#     relative_file_path = DIR_PATH + file_path
#     if (file_path.endswith("_1.csv")):
#         df1 = pd.read_csv(relative_file_path)
#         file2_path = relative_file_path[:-5] + "2.csv"
#         df2 = pd.read_csv(file2_path)

#         combined_df = pd.concat([df1, df2], ignore_index=True)
#         output_fie_path = relative_file_path[:-6] + ".csv"
#         combined_df.to_csv(output_fie_path, index=True)
#         print(f"CSV files appended and saved to: {output_fie_path}")
