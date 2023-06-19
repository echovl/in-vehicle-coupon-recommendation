import pandas as pd
from typing import cast
import seaborn as sns
from sklearn.model_selection import train_test_split

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

# Ingenieria de atributos

dataset["is_unemployed"] = dataset["occupation"].map(
    lambda o: 1 if o == "Unemployed" else 0)

dataset["is_student"] = dataset["occupation"].map(
    lambda o: 1 if o == "Student" else 0)

dataset.drop(columns=["occupation"], inplace=True)

# Encoding

categorical_columns = dataset.dtypes[dataset.dtypes ==
                                     "object"].index.to_list()
for column in categorical_columns:
    encoded = pd.get_dummies(dataset[column], prefix=column, dtype=int)

    print("encoding", column)
    # XGBoost necesita que los atributos no contengan los caracteres '[', ']' o '<'
    if "coupon_Restaurant(<20)" in encoded.columns:
       encoded.rename(
            {"coupon_Restaurant(<20)": "coupon_Restaurant(LessThan20)"}, axis=1, inplace=True)

    dataset.drop(columns=[column], inplace=True)
    dataset = dataset.join(encoded)
    
# Tabla de informacion final
dataset.head()

# Separamos la data, en variables independientes (x) y dependientes (y), para poder entrenar un árbol de clasificación
x = dataset.drop(["Y"], axis=1)
y = dataset["Y"]

# Mediante el método "train_test_split" usaremos el 20% de la data para probar el modelo. El parámetro "random state" nos sirve para
# poder replicar la misma separación
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)
