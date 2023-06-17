import pandas as pd

dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00603/in-vehicle-coupon-recommendation.csv"

dataset = pd.read_csv(dataset_url)

print(dataset.head())
