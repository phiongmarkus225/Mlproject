import os
import sys

from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from catboost import CatBoostRegressor

from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import (
    save_object,
    evaluate_models,
    optimize_model
)


@dataclass
class ModelTrainerConfig:

    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logging.info(
                "Splitting training and testing data."
            )

            # ==================================================
            # SPLIT TRAIN AND TEST ARRAY
            # ==================================================

            x_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            x_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # ==================================================
            # BASELINE MODELS
            # ==================================================

            models = {

                "Random Forest":
                    RandomForestRegressor(
                        random_state=42
                    ),

                "Gradient Boosting":
                    GradientBoostingRegressor(
                        random_state=42
                    ),

                "Linear Regression":
                    LinearRegression(),

                "K-Neighbors Regressor":
                    KNeighborsRegressor(
                        n_neighbors=5
                    ),

                "CatBoosting Regressor":
                    CatBoostRegressor(
                        verbose=False,
                        random_state=42
                    ),

                "AdaBoost Regressor":
                    AdaBoostRegressor(
                        random_state=42
                    ),

                "XGB Regressor":
                    XGBRegressor(
                        random_state=42,
                        objective="reg:squarederror"
                    )
            }

            # ==================================================
            # BASELINE MODEL EVALUATION
            # ==================================================

            logging.info(
                "Evaluating baseline models."
            )

            model_report = evaluate_models(
                x_train,
                y_train,
                x_test,
                y_test,
                models
            )

            logging.info(
                f"Baseline model report: {model_report}"
            )

            print("\n==============================")
            print("BASELINE MODEL RESULTS")
            print("==============================")

            for model_name, score in model_report.items():

                print(
                    f"{model_name}: R2 = {score:.4f}"
                )

            # ==================================================
            # FIND BASELINE BEST MODEL
            # ==================================================

            baseline_best_score = max(
                model_report.values()
            )

            baseline_best_model_name = max(
                model_report,
                key=model_report.get
            )

            print(
                f"\nBaseline Best Model: "
                f"{baseline_best_model_name}"
            )

            print(
                f"Baseline R2: "
                f"{baseline_best_score:.4f}"
            )

            # ==================================================
            # OPTUNA HYPERPARAMETER TUNING
            # ==================================================

            logging.info(
                "Starting Optuna hyperparameter tuning."
            )

            tuned_models = {}

            tuned_scores = {}

            best_params_dict = {}

            # ==================================================
            # TUNING EACH MODEL
            # ==================================================

            for model_name in models.keys():

                logging.info(
                    f"Starting Optuna tuning for "
                    f"{model_name}"
                )

                print(
                    f"\nTuning {model_name}..."
                )

                best_model, best_params, best_score = (
                    optimize_model(
                        model_name=model_name,
                        X_train=x_train,
                        y_train=y_train,
                        n_trials=30,
                        random_state=42
                    )
                )

                tuned_models[model_name] = best_model

                tuned_scores[model_name] = best_score

                best_params_dict[model_name] = best_params

                print(
                    f"Best Validation R2: "
                    f"{best_score:.4f}"
                )

                print(
                    f"Best Parameters: "
                    f"{best_params}"
                )

                logging.info(
                    f"{model_name} best parameters: "
                    f"{best_params}"
                )

            # ==================================================
            # EVALUATE TUNED MODELS ON TEST DATA
            # ==================================================

            logging.info(
                "Evaluating tuned models on test dataset."
            )

            tuned_model_report = evaluate_models(
                x_train,
                y_train,
                x_test,
                y_test,
                tuned_models
            )

            print("\n==============================")
            print("OPTUNA MODEL RESULTS")
            print("==============================")

            for model_name, score in tuned_model_report.items():

                print(
                    f"{model_name}: R2 = {score:.4f}"
                )

            logging.info(
                f"Tuned model report: "
                f"{tuned_model_report}"
            )

            # ==================================================
            # FIND BEST TUNED MODEL
            # ==================================================

            best_model_score = max(
                tuned_model_report.values()
            )

            best_model_name = max(
                tuned_model_report,
                key=tuned_model_report.get
            )

            best_model = tuned_models[
                best_model_name
            ]

            # ==================================================
            # CHECK MODEL PERFORMANCE
            # ==================================================

            if best_model_score < 0.6:

                raise CustomException(
                    "No best model found. "
                    f"Best R2 score: {best_model_score:.4f}"
                )

            # ==================================================
            # LOG BEST MODEL
            # ==================================================

            logging.info(
                "Best model found after Optuna tuning: "
                f"{best_model_name}"
            )

            logging.info(
                f"Best model R2 score: "
                f"{best_model_score:.4f}"
            )

            print("\n==============================")
            print("FINAL BEST MODEL")
            print("==============================")

            print(
                f"Model: {best_model_name}"
            )

            print(
                f"Test R2: {best_model_score:.4f}"
            )

            print(
                f"Parameters: "
                f"{best_params_dict[best_model_name]}"
            )

            # ==================================================
            # FINAL R2
            # ==================================================

            r2_square = r2_score(
                y_test,
                best_model.predict(x_test)
            )

            # ==================================================
            # SAVE MODEL
            # ==================================================

            logging.info(
                "Saving best model."
            )

            save_object(
                file_path=
                self.model_trainer_config
                .trained_model_file_path,

                obj=best_model
            )

            logging.info(
                "Best model saved successfully."
            )

            return r2_square

        except Exception as e:

            raise CustomException(
                e,
                sys
            )