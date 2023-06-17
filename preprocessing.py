import pandas as pd
from typing import cast
from utils import print_title

dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00603/in-vehicle-coupon-recommendation.csv"

# Cargamos el dataset
dataset = pd.read_csv(dataset_url)

print_title("Dataset")
print(dataset.head())

# Metricas basicas del dataset
print("Numero de muestras:", dataset.shape[0])
print("Numero de atributos", dataset.shape[1])
print("Tipos de atributos", dataset.dtypes)

# Analisis del dataset

# Calculamos el porcentaje de valores faltantes
missing_values = 100 * dataset.isna().sum() / len(dataset)

print_title("Valores faltantes")
print(missing_values)

# Calculamos la matrix de correlacion de nuestro dataset
corr = dataset.corr(numeric_only=True)

print_title("Matriz de correlacion")
print(corr)

# Preprocesamiento

# Eliminamos muestras duplicadas
dataset.drop_duplicates(inplace=True)

# Eliminamos los atributos 'car', 'direction_opp' y 'toCoupon_GEQ5min'
dataset.drop(columns=["car", "direction_opp",
             "toCoupon_GEQ5min"], inplace=True)

# Completamos los valores faltantes usando la moda de cada atributo
missing_values = cast(pd.Series, dataset.isna().sum())
missing_values = cast(pd.Series, missing_values[missing_values > 0])

for column in missing_values.to_dict():
    mode = dataset[column].value_counts().index[0]
    dataset[column].fillna(mode, inplace=True)

print("Existen valores faltantes?", dataset.isna().values.any())

# Ingenieria de atributos
dataset["is_unemployed"] = dataset["occupation"].map(
    lambda o: 1 if o == "Unemployed" else 0)

dataset["is_student"] = dataset["occupation"].map(
    lambda o: 1 if o == "Student" else 0)

dataset.drop(columns=["occupation"], inplace=True)

# One hot encoding
categorical_columns = dataset.dtypes[dataset.dtypes ==
                                     "object"].index.to_list()

for column in categorical_columns:
    encoded = pd.get_dummies(dataset[column], prefix=column, dtype=int)
    dataset.drop(columns=[column], inplace=True)
    dataset = dataset.join(encoded)

# Guardamos el dataset preprocesado
dataset.to_csv("in-vehicle-coupon-recommendation-processed.csv", index=False)
