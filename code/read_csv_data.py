import os
import csv
import urllib.request
import pandas as pd
import numpy as np

def retrieve_url(url, filename):
    if not os.path.exists(filename) and not os.path.isfile(filename):
        urllib.request.urlretrieve(url, filename)
    else:
        print(f'{filename} already exists! Nothing to download')

def get_csv(filename):
    csv_data = []
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            csv_data.append(row)
    return csv_data

if __name__ == '__main__':
    # dataset url https://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data'
    data_path = '../data'
    filename = f'{data_path}/concrete.csv'

    retrieve_url(url,filename)

    # method 1 to read csv data
    concrete_data = get_csv(filename)
    print(concrete_data[:5])

    concrete_data_np = np.array(concrete_data[1:], dtype=float)  # convert to numpy
    print(concrete_data_np[:5])

    # method 2 to read csv using pandas
    concrete_df = pd.read_csv(filename)
    print(concrete_df.head())

    concrete_df_np = concrete_df.to_numpy()  # convert to numpy
    print(concrete_df_np[:5])