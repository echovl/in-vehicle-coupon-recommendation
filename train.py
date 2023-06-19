import pandas as pd
from utils import print_title
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
import xgboost as xgb


def roc_auc_score(model, x, y):
    return metrics.roc_auc_score(y, model.predict_proba(x)[:, 1])


def train_stacking_model(estimators, x_train, y_train, x_test, y_test):
    stacking_model = StackingClassifier(estimators=estimators,
                                        final_estimator=LogisticRegression())
    stacking_model.fit(x_train, y_train)

    train_score = roc_auc_score(stacking_model, x_train, y_train)
    test_score = roc_auc_score(stacking_model, x_test, y_test)

    return train_score, test_score, stacking_model


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
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "sigmoid"],
            "gamma": ["scale", "auto"]
        }
        model = SVC(probability=True, random_state=33)
    elif name == "XGBoost":
        parameters = {
            "max_depth": [1, 5, 10, 50],
            "n_estimators": [100, 500, 1000, 2000]
        }
        model = xgb.XGBClassifier()
    elif name == "Logistic Regression":
        parameters = {
            "C": [0.01, 0.1, 1, 10, 100, 500]
        }
        model = LogisticRegression(random_state=0, C=1.0)
    elif name == "MLP":
        parameters = {
            "hidden_layer_sizes": [[100], [200], [300], [500]],
            "activation": ["identity", "logistic", "tanh", "relu"],
            "max_iter": [500, 700]
        }
        model = MLPClassifier(random_state=33)
    else:
        raise NotImplementedError()

    search = RandomizedSearchCV(
        estimator=model, param_distributions=parameters, scoring="roc_auc", n_jobs=-1, verbose=3)
    search.fit(x, y)
    best_params = search.best_params_

    if name == "Random Forest":
        model_tuned = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            max_features=best_params["max_features"],
            random_state=33
        )
    elif name == "SVM":
        model_tuned = SVC(
            C=best_params["C"],
            kernel=best_params["kernel"],
            gamma=best_params["gamma"],
            probability=True,
            random_state=33
        )
    elif name == "XGBoost":
        model_tuned = xgb.XGBClassifier(
            n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"])
    elif name == "Logistic Regression":
        model_tuned = LogisticRegression(random_state=33, C=best_params["C"])
    elif name == "MLP":
        model_tuned = MLPClassifier(
            hidden_layer_sizes=best_params["hidden_layer_sizes"],
            activation=best_params["activation"],
            max_iter=best_params["max_iter"]
        )
    else:
        raise NotImplementedError()

    model_tuned.fit(x_train, y_train)

    train_score = roc_auc_score(model_tuned, x_train, y_train)
    test_score = roc_auc_score(model_tuned, x_test, y_test)

    return train_score, test_score, model_tuned


dataset = pd.read_csv("./in-vehicle-coupon-recommendation-processed.csv")

x = dataset.drop("Y", axis=1)
y = dataset["Y"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=44)

models = ["Random Forest", "SVM", "XGBoost", "MLP"]
stacking_estimators = []

for model in models:
    print_title(model)

    train_score, test_score, model_tuned = train_model(
        model, x_train, y_train, x_test, y_test)

    print("Train AUC:", train_score)
    print("Test AUC:", test_score)

    stacking_estimators.append((model, model_tuned))

print_title("Stacking")
print(stacking_estimators)

train_score, test_score, model_tuned = train_stacking_model(
    stacking_estimators, x_train, y_train, x_test, y_test)

print("Train AUC:", train_score)
print("Test AUC:", test_score)
