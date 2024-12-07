import pandas as pd

def load_data(filename):
    data = pd.read_csv(filename)
    return data

def preprocess_data(data):
    # Handle missing values, outliers, etc.
    # ...
    return X, y