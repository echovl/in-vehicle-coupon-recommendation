import pandas as pd
from utils import print_title
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC


def roc_auc_score(model, x, y):
    return metrics.roc_auc_score(y, model.predict_proba(x)[:, 1])


def train_model(name, x_train, y_train, x_test, y_test):
    if name == "Random Forest":
        parameters = {
            "n_estimators": [500, 1000, 1500, 2000],
            "max_depth": [50, 100, 150, 200, 250, 300],
            "max_features": ["sqrt", "log2"]
        }
        model = RandomForestClassifier(random_state=33)
    elif name == "SVM":
        parameters = {
            "C": [0.1, 1, 10, 100, 1000],
            "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
            "gamma": ["scale", "auto"]
        }
        model = SVC(probability=True, random_state=33)
    else:
        raise NotImplementedError()

    search = RandomizedSearchCV(
        estimator=model, param_distributions=parameters, scoring="roc_auc", n_jobs=-1, verbose=3)
    search.fit(x, y)
    best_params = search.best_params_

    if name == "Random Forest":
        model_tuned = RandomForestClassifier(
            n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"], max_features=best_params["max_features"], random_state=33)
    elif name == "SVM":
        model_tuned = SVC(C=best_params["C"],
                          probability=True, random_state=33)
    else:
        raise NotImplementedError()

    model_tuned.fit(x_train, y_train)

    print("Train AUC:", roc_auc_score(model_tuned, x_train, y_train))
    print("Test AUC:", roc_auc_score(model_tuned, x_test, y_test))


dataset = pd.read_csv("./in-vehicle-coupon-recommendation-processed.csv")

x = dataset.drop("Y", axis=1)
y = dataset["Y"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=44)

print_title("Entrenando Modelos")


models = ["Random Forest", "SVM"]

for model in models:
    print_title(model)
    train_model(model, x_train, y_train, x_test, y_test)
