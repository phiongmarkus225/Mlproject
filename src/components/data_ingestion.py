"""Data ingestion module for the student performance ML project.

This module is responsible for reading the original dataset, saving a copy of
it into the artifacts folder, and splitting it into training and testing data.
The output files are used by the subsequent preprocessing and training steps.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    """Configuration for the data ingestion pipeline."""

    train_data_path: str = os.path.join(PROJECT_ROOT, 'artifacts', 'train.csv')
    test_data_path: str = os.path.join(PROJECT_ROOT, 'artifacts', 'test.csv')
    raw_data_path: str = os.path.join(PROJECT_ROOT, 'artifacts', 'data.csv')


class DataIngestion:
    """Load the dataset and prepare train/test files for model development."""

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self, data_source_path: str):
        """Read data from disk, save a raw copy, and split it into train/test sets.

        Parameters
        ----------
        data_source_path : str
            Path to the source CSV file that will be ingested.

        Returns
        -------
        tuple[str, str]
            Paths to the generated training and testing CSV files.
        """
        try:
            logging.info("Starting data ingestion process.")

            # Read the dataset from the provided source path
            df = pd.read_csv(data_source_path)
            logging.info(f"Dataset read successfully from {data_source_path}.")

            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            # Save the raw dataset to artifacts/data.csv
            df.to_csv(self.config.raw_data_path, index=False)
            logging.info(f"Raw data saved to {self.config.raw_data_path}.")

            # Split the dataset into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Dataset split into training and testing sets.")

            # Save the training and testing datasets
            train_set.to_csv(self.config.train_data_path, index=False)
            test_set.to_csv(self.config.test_data_path, index=False)
            logging.info(f"Training data saved to {self.config.train_data_path}.")
            logging.info(f"Testing data saved to {self.config.test_data_path}.")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion(config=DataIngestionConfig())
    train_path, test_path = obj.initiate_data_ingestion(data_source_path=os.path.join(PROJECT_ROOT, 'notebook', 'data', 'stud.csv'))
    
    data_transformation = DataTransformation()
    train_arr, test_arr,_= data_transformation.initiate_data_transformation(train_path, test_path)
    
    data_trainer = ModelTrainer()
    print(data_trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr)) # Replace None with actual data after transformation
