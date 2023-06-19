import pandas as pd
from typing import cast
import seaborn as sns

#Url Data
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00603/in-vehicle-coupon-recommendation.csv"

# Cargamos el dataset
dataset = pd.read_csv(dataset_url)

print("Dataset")
print(dataset.head())

# Metricas basicas del dataset
print("Numero de muestras:", dataset.shape[0])
print("Numero de atributos", dataset.shape[1])
print("Tipos de atributos", dataset.dtypes)

# Analisis del dataset

# Calculamos el porcentaje de valores faltantes
missing_values = 100 * dataset.isna().sum() / len(dataset)

print("Valores faltantes")
print(missing_values)

# Calculamos la matrix de correlacion de nuestro dataset
sns.heatmap(dataset.corr(), annot=True, cmap="crest")

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
#Tabla de informacion inicial
dataset.head()
