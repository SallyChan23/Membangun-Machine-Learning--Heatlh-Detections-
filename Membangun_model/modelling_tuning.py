import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import mlflow
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing.automate_Jeselyn_Tania import load_and_preprocess
import dagshub
dagshub.init(repo_owner='sallychan23',
             repo_name='health-model-jeselyn',
             mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

X_train, X_test, y_train, y_test = load_and_preprocess("preprocessing/healthcare_preprocessed.csv")

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10],
}

for n in param_grid["n_estimators"]:
    for d in param_grid["max_depth"]:
        with mlflow.start_run():
            model = XGBClassifier(
                n_estimators=n,
                max_depth=d,
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=42
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average='macro', zero_division=0)
            rec = recall_score(y_test, preds, average='macro', zero_division=0)
            f1 = f1_score(y_test, preds, average='macro', zero_division=0)

            mlflow.log_param("n_estimators", n)
            mlflow.log_param("max_depth", d)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            print(f"[{n} estimators | depth {d}] acc={acc:.3f}, f1={f1:.3f}")
