"""Prediction pipeline for serving single predictions.

This module is used by the web application to:
1. collect user input as a data class,
2. build a DataFrame matching the training schema,
3. apply the saved preprocessor,
4. run the trained model,
5. return the predicted math score.
"""

import sys
import os
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object


class CustomData:
    """Container for the features a user submits through the web form.

    The attribute names MUST match the column names used during training
    (see data_transformation.py) so the preprocessor can transform them.
    """

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """Return the input as a single-row DataFrame with training columns."""
        try:
            return pd.DataFrame(
                [{
                    "gender": self.gender,
                    "race_ethnicity": self.race_ethnicity,
                    "parental_level_of_education": self.parental_level_of_education,
                    "lunch": self.lunch,
                    "test_preparation_course": self.test_preparation_course,
                    "reading_score": self.reading_score,
                    "writing_score": self.writing_score,
                }]
            )
        except Exception as e:
            raise CustomException(e, sys)


class PredictPipeline:
    """Load the saved artifacts and run a prediction."""

    def __init__(self):
        self.model_path = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")
        self.preprocessor_path = os.path.join(PROJECT_ROOT, "artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame) -> float:
        """Transform raw features and return the predicted math score.

        Parameters
        ----------
        features : pd.DataFrame
            Single-row DataFrame with the same columns used in training.

        Returns
        -------
        float
            Predicted math score.
        """
        try:
            logging.info("Loading preprocessor and model artifacts.")

            preprocessor = load_object(file_path=self.preprocessor_path)
            model = load_object(file_path=self.model_path)

            logging.info("Applying preprocessing to input features.")
            transformed_features = preprocessor.transform(features)

            logging.info("Running model prediction.")
            prediction = model.predict(transformed_features)

            return prediction[0]

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Quick smoke test so you can verify the prediction pipeline works
    # before wiring it into the web app.
    sample = CustomData(
        gender="female",
        race_ethnicity="group B",
        parental_level_of_education="bachelor's degree",
        lunch="standard",
        test_preparation_course="none",
        reading_score=72,
        writing_score=74,
    )

    pipe = PredictPipeline()
    result = pipe.predict(sample.get_data_as_data_frame())
    print(f"\n[OK] Predicted math score: {result:.2f}")