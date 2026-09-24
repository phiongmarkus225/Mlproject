"""Training pipeline orchestration.

This module ties together the three core components:
data ingestion -> data transformation -> model training.
Run this pipeline to (re)train the model from raw data.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


class TrainPipeline:
    """High-level orchestrator for the training workflow."""

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()

    def run(self, data_source_path: str) -> float:
        """Execute the full training flow.

        Parameters
        ----------
        data_source_path : str
            Path to the raw CSV dataset.

        Returns
        -------
        float
            R2 score of the best model on the held-out test set.
        """
        try:
            logging.info("=== Starting training pipeline ===")

            # 1. Ingestion
            data_ingestion = DataIngestion(config=self.data_ingestion_config)
            train_path, test_path = data_ingestion.initiate_data_ingestion(
                data_source_path=data_source_path
            )

            # 2. Transformation
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = (
                data_transformation.initiate_data_transformation(
                    train_path=train_path,
                    test_path=test_path
                )
            )

            # 3. Model training
            model_trainer = ModelTrainer()
            r2_score = model_trainer.initiate_model_trainer(
                train_array=train_arr,
                test_array=test_arr
            )

            logging.info("=== Training pipeline completed ===")
            return r2_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Default source data lives in the notebook folder.
    default_source = os.path.join(
        PROJECT_ROOT, "notebook", "data", "stud.csv"
    )

    source = sys.argv[1] if len(sys.argv) > 1 else default_source

    pipeline = TrainPipeline()
    score = pipeline.run(data_source_path=source)
    print(f"\n[OK] Training finished. Best test R2 = {score:.4f}")