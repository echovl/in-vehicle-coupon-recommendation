##Librerías necesarias

import pandas as pd
from typing import cast
import seaborn as sns
from sklearn.model_selection import train_test_split

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
sns.heatmap(dataset.corr(numeric_only=True), annot=True, cmap="crest")

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

# Clasificador RandomForest con los mejores hiperparámetros
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

# clasificador SVC
svm_model = SVC()
# Definición de hiperparámetros a buscar en la búsqueda aleatoria
svm_parameters = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf", "sigmoid"],
    "gamma": ["scale", "auto"]
}
# Búsqueda aleatoria de hiperparámetros
svm_model_search = RandomizedSearchCV(
    svm_model,
    param_distributions=svm_parameters,
    n_iter=10,  # Número de combinaciones de hiperparámetros a probar
    cv=5,  # Número de divisiones para la validación cruzada
    random_state=42,
    n_jobs=-1,
    verbose=2
)
svm_model_search.fit(x_train, y_train)

# Clasificador SVC con los mejores hiperparámetros
svm_model = SVC(**svm_model_search.best_params_, probability=True)

# Entrenamineto del modelo
svm_model.fit(x_train, y_train)
# Cálculo del AUC y gráfica de la Curva ROC
plot_roc_curve(svm_model, x_test, y_test)
# Evaluar el modelo con los datos de prueba
y_pred = svm_model.predict(x_test)

#AUNQUE SE HA DEFINIDO EL AUC COMO MÉTRICA
# se imprime un resumen de los demás indicadores
print(classification_report(y_test, y_pred))

# Crear el clasificador XGBoost
xgb_model = xgb.XGBClassifier()

# Definir los hiperparámetros a buscar en la búsqueda aleatoria
xgb_parameters = {
    "max_depth": [1, 5, 10, 50],
    "n_estimators": [100, 500, 1000, 2000]
}

# Realizar la búsqueda aleatoria de hiperparámetros
xgb_model_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=xgb_parameters,
    n_iter=10,  # Número de combinaciones de hiperparámetros a probar
    cv=5,  # Número de divisiones para la validación cruzada
    random_state=42,
    n_jobs=-1,
    verbose=2
)
xgb_model_search.fit(x_train, y_train)

# Clasificador XGBBoost con los mejores hiperparámetros
xgb_model = xgb.XGBClassifier(**xgb_model_search.best_params_)

# Entrenamiento del modelo
xgb_model.fit(x_train, y_train)

# Cálculo del AUC y gráfica de la Curva ROC
plot_roc_curve(xgb_model, x_test, y_test)

# Evaluar el modelo con los datos de prueba
y_pred = xgb_model.predict(x_test)

#AUNQUE SE HA DEFINIDO EL AUC COMO MÉTRICA
# se imprime un resumen de los demás indicadores
print(classification_report(y_test, y_pred))

# Clasificador de redes neuronales
mlp_model = MLPClassifier()

# Definición de hiperparámetros a buscar en la búsqueda aleatoria
mlp_parameters = {
    "hidden_layer_sizes": [[100], [200], [300], [500]],
    "activation": ["identity", "logistic", "tanh", "relu"],
    "max_iter": [700, 700]
}

# Búsqueda aleatoria de hiperparámetros
mlp_model_search = RandomizedSearchCV(
    mlp_model,
    param_distributions=mlp_parameters,
    n_iter=10,  # Número de combinaciones de hiperparámetros a probar
    cv=5,  # Número de divisiones para la validación cruzada
    random_state=42,
    n_jobs=-1
)
mlp_model_search.fit(x_train, y_train)

# Clasificador LogisticRegression con los mejores hiperparámetros
mlp_model = MLPClassifier(**mlp_model_search.best_params_)

# Entrenamineto del modelo
mlp_model.fit(x_train, y_train)

# Cálculo del AUC y gráfica de la Curva ROC
plot_roc_curve(mlp_model, x_test, y_test)

# Evaluar el modelo con los datos de prueba
y_pred = mlp_model.predict(x_test)

#AUNQUE SE HA DEFINIDO EL AUC COMO MÉTRICA
# se imprime un resumen de los demás indicadores
print(classification_report(y_test, y_pred))

# Utilizamos nuestros clasificadores optimizados
estimators = [
    ("RandomForest", rf_model),
    ("XGBoost", xgb_model),
    ("MLP", mlp_model)
]

# Creamos el modelo ensamble y realizamos el entrenamiento
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(penalty="l2"))

stacking_model.fit(x_train, y_train)

# Cálculo del AUC y gráfica de la Curva ROC
plot_roc_curve(stacking_model, x_test, y_test)

# Evaluar el modelo con los datos de prueba
y_pred = stacking_model.predict(x_test)

# Imprimir informe de clasificación
print(classification_report(y_test, y_pred))


# Comparativa AUC
models = [
  lr_model,
  rf_model,
  svm_model,
  xgb_model,
  mlp_model,
  stacking_model
]

colors = rainbow(np.linspace(0, 1, len(models)))

plt.figure(figsize = (20, 12))
plt.plot([0,1], [0,1], 'r--')

for (idx, model) in enumerate(models):
  model_name = type(model).__name__
  y_pred_proba = model.predict_proba(x_test)[::, 1]
  fpr, tpr, th = roc_curve(y_test,  y_pred_proba)
  auc = roc_auc_score(y_test, y_pred_proba)

  label = model_name + " AUC:" + " {0:.3f}".format(auc)
  plt.plot(fpr, tpr, c = colors[idx], label = label, linewidth = 4)

plt.xlabel("False Positive Rate", fontsize = 16)
plt.ylabel("True Positive Rate", fontsize = 16)
plt.title("Analisis comparativo de la métrica AUC", fontsize = 16)
plt.legend(loc = "lower right", fontsize = 16)
