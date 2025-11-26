import os
import sys
import numpy as np
import pandas as pd

"""
Defining common constant variable for training pipeline
"""

TARGET_COLUMN = "Failure Type"
PIPELINE_NAME: str = "MachineMaintenance"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "data.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

FINAL_MODEL_DIR="final_model"
MODEL_FILE_NAME = "model.pkl"

"""
Data Ingestion related constant variable start with DATA_INGESTION 
"""
DATA_INGESTION_COLLECTION_NAME: str = "machineData"
DATA_INGESTION_DATABASE_NAME: str = "MACHINE"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

