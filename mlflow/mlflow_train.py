import sys

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm

import mlflow
import mlflow.sklearn

# load data
data = pd.read_csv("mlflow/data2/aug_train.csv")
targets = data[["target"]]


data.drop(["enrollee_id", "target"], inplace=True, axis=1)

# process features
## fill in missing categorical variables and label encode them

categorical_features = []
numerical_features = []

# distinguish the different features
for column in data.columns:
    dtype = str(data[column].dtype)
    if dtype in ["float64", "int64"]:
        numerical_features.append(column)
    else:
        categorical_features.append(column)

for categorical_feature in categorical_features:
    data[categorical_feature].fillna("missing", inplace=True)
    le = LabelEncoder()
    data[categorical_feature] = le.fit_transform(data[categorical_feature])

print("features processed")

# split train / test

x_train, x_test, y_train, y_test = train_test_split(
    data.values,
    targets.values.ravel(),
    test_size=0.3,
    random_state=2021,
    stratify=targets.values,
)
alpha = sys.argv[0] if len(sys.argv) > 1 else 0.5


# experiment_id = mlflow.create_experiment("training experiment")
mlflow.set_experiment("training experiment")
experiment = mlflow.get_experiment_by_name("training experiment")
experiment_id  = experiment.experiment_id
# experiment_id = mlflow.set_experiment("mlflow experiment")


n_estimators_range = np.arange(10, 20, 30)
max_depth_range = np.arange(1, 10, 2)
max_features_range = ["sqrt", None, "log2"]


for n_estimators in tqdm(n_estimators_range):
    for max_depth in tqdm(max_depth_range, leave=False):
        for max_features in tqdm(max_features_range, leave=False):

            with mlflow.start_run(experiment_id=experiment_id):

                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    n_jobs=3,
                )

                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred)

                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("max_features", max_features)

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("auc", auc)

                mlflow.sklearn.log_model(model, "model") # record sklearn model