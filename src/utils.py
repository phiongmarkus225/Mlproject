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


def load_object(file_path):
    """Load a serialized object (model or preprocessor) from disk.

    Parameters
    ----------
    file_path : str
        Path to the serialized object.

    Returns
    -------
    object
        The deserialized Python object.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Artifact not found: {file_path}")

        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

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
    

"""Utility helpers for the ML project.

This module contains reusable functions for:
- Saving Python objects
- Evaluating multiple regression models
- Hyperparameter tuning using Optuna
"""


import optuna

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException

def optimize_model(
    model_name,
    X_train,
    y_train,
    n_trials=30,
    random_state=42
):
    """
    Hyperparameter tuning using Optuna.

    Parameters
    ----------
    model_name : str
        Name of the regression model.

    X_train : array-like
        Training features.

    y_train : array-like
        Training target.

    n_trials : int
        Number of Optuna trials.

    random_state : int
        Random state.

    Returns
    -------
    best_model : object
        Model with the best hyperparameters.

    best_params : dict
        Best hyperparameters found by Optuna.

    best_score : float
        Best validation R2 score.
    """

    try:

        # --------------------------------------------------
        # Split training data into training and validation
        # --------------------------------------------------

        X_train_optuna, X_valid, y_train_optuna, y_valid = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=random_state
        )

        # --------------------------------------------------
        # Objective Function
        # --------------------------------------------------

        def objective(trial):

            # ==================================================
            # RANDOM FOREST
            # ==================================================

            if model_name == "Random Forest":

                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators",
                        100,
                        500
                    ),

                    "max_depth": trial.suggest_int(
                        "max_depth",
                        3,
                        30
                    ),

                    "min_samples_split": trial.suggest_int(
                        "min_samples_split",
                        2,
                        20
                    ),

                    "min_samples_leaf": trial.suggest_int(
                        "min_samples_leaf",
                        1,
                        10
                    ),

                    "max_features": trial.suggest_categorical(
                        "max_features",
                        ["sqrt", "log2", 1.0]
                    ),

                    "random_state": random_state,

                    "n_jobs": -1
                }

                model = RandomForestRegressor(**params)

            # ==================================================
            # GRADIENT BOOSTING
            # ==================================================

            elif model_name == "Gradient Boosting":

                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators",
                        100,
                        500
                    ),

                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        0.01,
                        0.3,
                        log=True
                    ),

                    "max_depth": trial.suggest_int(
                        "max_depth",
                        2,
                        10
                    ),

                    "min_samples_split": trial.suggest_int(
                        "min_samples_split",
                        2,
                        20
                    ),

                    "min_samples_leaf": trial.suggest_int(
                        "min_samples_leaf",
                        1,
                        10
                    ),

                    "subsample": trial.suggest_float(
                        "subsample",
                        0.5,
                        1.0
                    ),

                    "random_state": random_state
                }

                model = GradientBoostingRegressor(**params)

            # ==================================================
            # KNN
            # ==================================================

            elif model_name == "K-Neighbors Regressor":

                params = {
                    "n_neighbors": trial.suggest_int(
                        "n_neighbors",
                        2,
                        30
                    ),

                    "weights": trial.suggest_categorical(
                        "weights",
                        ["uniform", "distance"]
                    ),

                    "p": trial.suggest_int(
                        "p",
                        1,
                        2
                    )
                }

                model = KNeighborsRegressor(**params)

            # ==================================================
            # CATBOOST
            # ==================================================

            elif model_name == "CatBoosting Regressor":

                params = {
                    "iterations": trial.suggest_int(
                        "iterations",
                        100,
                        1000
                    ),

                    "depth": trial.suggest_int(
                        "depth",
                        4,
                        10
                    ),

                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        0.01,
                        0.3,
                        log=True
                    ),

                    "l2_leaf_reg": trial.suggest_float(
                        "l2_leaf_reg",
                        1,
                        10
                    ),

                    "random_seed": random_state,

                    "verbose": False
                }

                model = CatBoostRegressor(**params)

            # ==================================================
            # ADABOOST
            # ==================================================

            elif model_name == "AdaBoost Regressor":

                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators",
                        50,
                        500
                    ),

                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        0.01,
                        1.0,
                        log=True
                    ),

                    "loss": trial.suggest_categorical(
                        "loss",
                        [
                            "linear",
                            "square",
                            "exponential"
                        ]
                    ),

                    "random_state": random_state
                }

                model = AdaBoostRegressor(**params)

            # ==================================================
            # XGBOOST
            # ==================================================

            elif model_name == "XGB Regressor":

                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators",
                        100,
                        1000
                    ),

                    "max_depth": trial.suggest_int(
                        "max_depth",
                        3,
                        12
                    ),

                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        0.01,
                        0.3,
                        log=True
                    ),

                    "subsample": trial.suggest_float(
                        "subsample",
                        0.5,
                        1.0
                    ),

                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree",
                        0.5,
                        1.0
                    ),

                    "min_child_weight": trial.suggest_int(
                        "min_child_weight",
                        1,
                        10
                    ),

                    "gamma": trial.suggest_float(
                        "gamma",
                        0,
                        10
                    ),

                    "reg_alpha": trial.suggest_float(
                        "reg_alpha",
                        0,
                        10
                    ),

                    "reg_lambda": trial.suggest_float(
                        "reg_lambda",
                        1,
                        10
                    ),

                    "random_state": random_state,

                    "n_jobs": -1,

                    "objective": "reg:squarederror"
                }

                model = XGBRegressor(**params)

            # ==================================================
            # LINEAR REGRESSION
            # ==================================================

            elif model_name == "Linear Regression":

                params = {
                    "fit_intercept": trial.suggest_categorical(
                        "fit_intercept",
                        [True, False]
                    ),

                    "positive": trial.suggest_categorical(
                        "positive",
                        [True, False]
                    )
                }

                model = LinearRegression(**params)

            else:

                raise ValueError(
                    f"Unknown model: {model_name}"
                )

            # --------------------------------------------------
            # Training
            # --------------------------------------------------

            model.fit(
                X_train_optuna,
                y_train_optuna
            )

            # --------------------------------------------------
            # Validation prediction
            # --------------------------------------------------

            y_valid_pred = model.predict(X_valid)

            # --------------------------------------------------
            # R2 Score
            # --------------------------------------------------

            score = r2_score(
                y_valid,
                y_valid_pred
            )

            return score

        # --------------------------------------------------
        # Create Optuna Study
        # --------------------------------------------------

        study = optuna.create_study(
            direction="maximize",
            study_name=f"{model_name}_optimization"
        )

        # --------------------------------------------------
        # Run Optuna
        # --------------------------------------------------

        study.optimize(
            objective,
            n_trials=n_trials
        )

        # --------------------------------------------------
        # Get Best Parameters
        # --------------------------------------------------

        best_params = study.best_params

        # --------------------------------------------------
        # Create Final Model
        # --------------------------------------------------

        if model_name == "Random Forest":

            best_model = RandomForestRegressor(
                **best_params,
                random_state=random_state,
                n_jobs=-1
            )

        elif model_name == "Gradient Boosting":

            best_model = GradientBoostingRegressor(
                **best_params,
                random_state=random_state
            )

        elif model_name == "Linear Regression":

            best_model = LinearRegression(
                **best_params
            )

        elif model_name == "K-Neighbors Regressor":

            best_model = KNeighborsRegressor(
                **best_params
            )

        elif model_name == "CatBoosting Regressor":

            best_model = CatBoostRegressor(
                **best_params,
                verbose=False,
                random_seed=random_state
            )

        elif model_name == "AdaBoost Regressor":

            best_model = AdaBoostRegressor(
                **best_params,
                random_state=random_state
            )

        elif model_name == "XGB Regressor":

            best_model = XGBRegressor(
                **best_params,
                random_state=random_state,
                n_jobs=-1,
                objective="reg:squarederror"
            )

        # --------------------------------------------------
        # Train best model on ALL training data
        # --------------------------------------------------

        best_model.fit(
            X_train,
            y_train
        )

        return (
            best_model,
            best_params,
            study.best_value
        )
        

    except Exception as e:

        raise CustomException(e, sys)