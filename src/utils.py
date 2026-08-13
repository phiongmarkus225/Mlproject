"""Utility helpers for the ML project.

This module contains small reusable functions that support the training pipeline,
including saving Python objects such as fitted preprocessors or models.
"""

import sys
import os
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import CustomException
import dill


def save_object(file_path, obj):
    """Serialize and save an object to disk using dill.

    Parameters
    ----------
    file_path : str
        Destination path where the object will be stored.
    obj : object
        Python object to serialize, such as a fitted preprocessing pipeline.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    """Evaluate multiple regression models and return their R^2 scores.

    Parameters
    ----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training target values.
    X_test : array-like
        Testing features.
    y_test : array-like
        Testing target values.
    models : dict
        Dictionary of model names and their corresponding instantiated objects.

    Returns
    -------
    dict
        A dictionary containing model names as keys and their R^2 scores on the test set as values.
    """
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)