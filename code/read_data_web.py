import os
import tensorflow as tf
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


iris_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

def retrieve_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    cache_dir = '..'
    cache_subdir = 'data'
    filepath = tf.keras.utils.get_file('iris.data', url, cache_dir=cache_dir, cache_subdir=cache_subdir)

    return filepath

def parse_data(filepath):

    df = pd.read_csv(filepath, names=iris_columns)
    print(df.head())

    df['species'].replace(label_map, inplace=True)
    return df

def convert_data_to_dataset(df):
    features = df[iris_columns[:4]]
    labels = df[iris_columns[-1]]

    iris_dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    return iris_dataset

if __name__ == '__main__':

    filepath = retrieve_data()

    df = parse_data(filepath)

    iris_dataset = convert_data_to_dataset(df)

    for example in iris_dataset.take(5):
        print(iris_dataset)
