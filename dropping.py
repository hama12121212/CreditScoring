import pandas as pd
import numpy as np



POS_CASH_balance = pd.read_csv('C:/Users/Barao/Downloads/New folder/POS_CASH_balance.csv')


def drop_rows_with_excessive_missing(data, threshold):
    # Compute the ratio of missing values in each row
    missing_ratio = data.isnull().sum(axis=1) / data.shape[1]
    
    # Filter rows that exceed the threshold
    rows_to_drop = missing_ratio[missing_ratio > threshold].index
    
    # Drop rows from the DataFrame
    data_dropped = data.drop(rows_to_drop)
    
    return data_dropped


threshold = 0.3


POS_CASH_balance_dropped = drop_rows_with_excessive_missing(POS_CASH_balance, threshold)


import csv

output_file_path = "C:/Users/Barao/Downloads/POS_CASH_balance_dropped.csv"

# Convert the DataFrame to a list of lists
data_to_write = POS_CASH_balance_dropped.values.tolist()

# Write the data to the CSV file
with open(output_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_to_write)