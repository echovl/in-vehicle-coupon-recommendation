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


# Entrenamiento de Modelos
##Librerías necesarias
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import log_loss,roc_auc_score
from sklearn.metrics import roc_curve, auc
from matplotlib.cm import rainbow

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
import xgboost as xgb

from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


# Creación de una función para el ploteo de la curva AUC
def plot_roc_curve(model, x_test, y_test):
  y_pred_proba = model.predict_proba(x_test)[::,1]
  fpr, tpr, th = roc_curve(y_test,  y_pred_proba)
  auc = roc_auc_score(y_test, y_pred_proba)
  model_name = type(model).__name__
  plt.title(f"{model_name}'s ROC Curve")
  plt.plot(fpr,tpr,label="auc="+str(auc))
  plt.xlabel("FPR")
  plt.ylabel("TPR")
  plt.legend(loc=4)
  plt.show()

 ## Regresion Logistica

# Creación del clasificador de regresión logistica
lr_model = LogisticRegression()

# Definición de los hiperparámetros a buscar en la búsqueda aleatoria
lr_parameters = {
    "C":[0.01, 0.1, 1, 10, 100, 500],
    "max_iter": [500, 1000]
}

# Búsqueda aleatoria de hiperparámetros
lr_model_search = RandomizedSearchCV(
    lr_model,
    param_distributions=lr_parameters,
    n_iter=10,  # Número de combinaciones de hiperparámetros a probar
    cv=5,  # Número de divisiones para la validación cruzada
    random_state=42,
    n_jobs=-1,
    verbose=2
)
lr_model_search.fit(x_train, y_train)

# Creacion del clasificador LogisticRegression con los mejores hiperparámetros
lr_model = LogisticRegression(**lr_model_search.best_params_)

# Entrenamiento del modelo con los datos de entrenamiento
lr_model.fit(x_train, y_train)

# Cálculo del AUC y gráfica de la Curva ROC
plot_roc_curve(lr_model, x_test, y_test)

# Evaluar el modelo con los datos de prueba
y_pred = lr_model.predict(x_test)

#AUNQUE SE HA DEFINIDO EL AUC COMO MÉTRICA
# se imprime un resumen de los demás indicadores
print(classification_report(y_test, y_pred))

## Random Forest

# Clasificador Random Forest
rf_model = RandomForestClassifier()

# Definición de hiperparámetros a buscar en la búsqueda aleatoria
rf_parameters = {
    "n_estimators": [500, 1000, 1500, 2000],
    "max_depth": [50, 100, 150, 200, 250, 300],
    "max_features": ["sqrt", "log2"]
}

# Búsqueda aleatoria de hiperparámetros
rf_model_search = RandomizedSearchCV(
    rf_model,
    param_distributions=rf_parameters,
    n_iter=10,  # Número de combinaciones de hiperparámetros a probar
    cv=5,  # Número de divisiones para la validación cruzada
    random_state=42,
    n_jobs=-1,
    verbose=2
)
rf_model_search.fit(x_train, y_train)

# Clasificador LogisticRegression con los mejores hiperparámetros
rf_model = RandomForestClassifier(**rf_model_search.best_params_)

# Entrenamineto del modelo
rf_model.fit(x_train, y_train)

# Cálculo del AUC y gráfica de la Curva ROC
plot_roc_curve(rf_model, x_test, y_test)

# Evaluar el modelo con los datos de prueba
y_pred = rf_model.predict(x_test)

#AUNQUE SE HA DEFINIDO EL AUC COMO MÉTRICA
# se imprime un resumen de los demás indicadores
print(classification_report(y_test, y_pred))

